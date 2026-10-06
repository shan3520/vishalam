import torch

from vishalam.model.config import ModelConfig
from vishalam.model.model import Vishalam
from vishalam.train.ckpt import load_checkpoint, save_checkpoint
from vishalam.train.muon import Muon, newton_schulz
from vishalam.train.pretrain import build_optimizers

TINY = ModelConfig(vocab_size=300, d_model=64, n_layers=2, n_heads=4, ffn_dim=128, max_seq_len=64)
MUON_CFG = {"optimizer": "muon", "lr": 3e-3, "weight_decay": 0.1, "muon_lr": 0.02, "muon_momentum": 0.95}


def test_newton_schulz_orthogonalizes():
    torch.manual_seed(0)
    for shape in [(64, 32), (32, 64), (96, 48)]:
        G = torch.randn(*shape) * 7.0  # scale must not matter: the iteration normalizes first
        s = torch.linalg.svdvals(newton_schulz(G))
        # the tuned quintic iteration lands singular values in a band around 1, not exactly on 1
        assert s.max() < 1.3 and s.min() > 0.5, (shape, s.min().item(), s.max().item())


def test_param_split_between_muon_and_adamw():
    m = Vishalam(TINY)
    muon, adam = build_optimizers(m, MUON_CFG, "cpu")
    assert isinstance(muon, Muon) and isinstance(adam, torch.optim.AdamW)
    muon_ids = {id(p) for g in muon.param_groups for p in g["params"]}
    adam_ids = {id(p) for g in adam.param_groups for p in g["params"]}
    all_ids = {id(p) for p in m.parameters()}
    assert muon_ids.isdisjoint(adam_ids) and muon_ids | adam_ids == all_ids
    assert id(m.embed.weight) in adam_ids  # embedding (tied head) stays on AdamW
    assert any(g["split"] == 3 for g in muon.param_groups)  # fused q/k/v orthogonalized per block
    assert all("base_lr" in g for o in (muon, adam) for g in o.param_groups)


def test_muon_trains_tiny_model():
    torch.manual_seed(0)
    m = Vishalam(TINY)
    opts = build_optimizers(m, MUON_CFG, "cpu")
    x = torch.randint(0, 300, (4, 33))
    for _ in range(120):
        _, loss = m(x[:, :-1], x[:, 1:])
        for o in opts:
            o.zero_grad()
        loss.backward()
        for o in opts:
            o.step()
    assert loss.item() < 0.5, loss.item()


def test_checkpoint_formats(tmp_path):
    torch.manual_seed(0)
    m = Vishalam(TINY)
    # two optimizers round-trip
    opts = build_optimizers(m, MUON_CFG, "cpu")
    _, loss = m(torch.randint(0, 300, (2, 8)), torch.randint(0, 300, (2, 8)))
    loss.backward()
    for o in opts:
        o.step()
    save_checkpoint(tmp_path / "c.pt", m, opts, None, 3, {})
    opts2 = build_optimizers(Vishalam(TINY), MUON_CFG, "cpu")
    load_checkpoint(tmp_path / "c.pt", Vishalam(TINY), opts2)
    assert len(opts2[0].state) == len(opts[0].state) > 0
    # an old single-AdamW checkpoint loads into the new list API, keeping base_lr
    adam_only = build_optimizers(m, {"lr": 1e-3, "weight_decay": 0.1}, "cpu")
    save_checkpoint(tmp_path / "old.pt", m, adam_only[0], None, 5, {})  # old call style
    fresh = build_optimizers(m, {"lr": 1e-3, "weight_decay": 0.1}, "cpu")
    assert load_checkpoint(tmp_path / "old.pt", m, fresh)["step"] == 5
    assert all(g["base_lr"] == 1e-3 for g in fresh[0].param_groups)


def test_gpu_checkpoint_loads_without_its_fused_adamw_flag(tmp_path):
    """A GPU session saves AdamW(fused=True); a TPU/CPU session loading it must keep its own
    implementation (on XLA the fused CUDA path crashes)."""
    torch.manual_seed(0)
    gpu = Vishalam(TINY)
    gpu_opts = [torch.optim.AdamW(gpu.parameters(), lr=1e-3, fused=True)]
    x = torch.randint(0, TINY.vocab_size, (2, 8))
    gpu(x, x)[1].backward()
    gpu_opts[0].step()
    save_checkpoint(tmp_path / "ckpt.pt", gpu, gpu_opts, None, 1, {})

    tpu = Vishalam(TINY)
    tpu_opts = [torch.optim.AdamW(tpu.parameters(), lr=1e-3, foreach=True)]
    load_checkpoint(tmp_path / "ckpt.pt", tpu, tpu_opts)
    g = tpu_opts[0].param_groups[0]
    assert not g["fused"] and g["foreach"]
    state = next(iter(tpu_opts[0].state.values()))
    assert state["step"].device.type == "cpu" and float(state["step"]) == 1.0
    tpu(x, x)[1].backward()
    tpu_opts[0].step()  # the non-fused path runs with the loaded state
