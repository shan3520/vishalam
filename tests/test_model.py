import torch

from vishalam.model.config import ModelConfig, model_config_from_yaml
from vishalam.model.model import KVCache, Vishalam

TINY = ModelConfig(vocab_size=300, d_model=64, n_layers=2, n_heads=4, ffn_dim=128, max_seq_len=64)


def test_param_counts_match_plan():
    sizes = {n: Vishalam(model_config_from_yaml(f"configs/vishalam-{n}.yaml")).num_params() / 1e6
             for n in ["30m", "125m"]}
    assert 25 < sizes["30m"] < 35
    assert 115 < sizes["125m"] < 135


def test_350m_param_count_without_allocating():
    cfg = model_config_from_yaml("configs/vishalam-350m.yaml")
    with torch.device("meta"):
        n = Vishalam(cfg).num_params() / 1e6
    assert 340 < n < 380


def test_forward_shapes_and_loss():
    torch.manual_seed(0)
    m = Vishalam(TINY)
    x = torch.randint(0, 300, (2, 16))
    y = torch.randint(0, 300, (2, 16))  # independent targets (tied embeddings favour y == x)
    logits, loss = m(x, y, return_logits=True)
    assert logits.shape == (2, 16, 300)
    assert abs(loss.item() - torch.log(torch.tensor(300.0)).item()) < 0.5  # ~uniform at init


def test_loss_mask():
    torch.manual_seed(0)
    m = Vishalam(TINY)
    x = torch.randint(0, 300, (1, 16))
    mask = torch.zeros(1, 16)
    mask[0, 8:] = 1
    _, l_masked = m(x, x, loss_mask=mask)
    logits, _ = m(x, x, return_logits=True)
    ce = torch.nn.functional.cross_entropy(logits[0, 8:], x[0, 8:])
    assert torch.allclose(l_masked, ce, atol=1e-5)


def test_causality():
    torch.manual_seed(0)
    m = Vishalam(TINY).eval()
    x = torch.randint(0, 300, (1, 12))
    y = x.clone()
    y[0, 8:] = (y[0, 8:] + 1) % 300
    with torch.no_grad():
        a, _ = m(x, x, return_logits=True)
        b, _ = m(y, y, return_logits=True)
    assert torch.allclose(a[0, :8], b[0, :8], atol=1e-5)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    m = Vishalam(TINY).eval()
    x = torch.randint(0, 300, (1, 20))
    with torch.no_grad():
        full, _ = m(x, x, return_logits=True)
        cache = KVCache(TINY, 1, 64, "cpu", torch.float32)
        m(x[:, :13], cache=cache)  # prefill 13 tokens
        step_logits = []
        for t in range(13, 20):
            lg, _ = m(x[:, t : t + 1], cache=cache)
            step_logits.append(lg[0, -1])
        # multi-token chunk after a prefill (exercises the offset mask)
        cache2 = KVCache(TINY, 1, 64, "cpu", torch.float32)
        m(x[:, :10], cache=cache2)
        chunk, _ = m(x[:, 10:20], cache=cache2)
    for i, t in enumerate(range(13, 20)):
        assert torch.allclose(step_logits[i], full[0, t], atol=1e-4)
    assert torch.allclose(chunk[0, -1], full[0, 19], atol=1e-4)


def test_overfit_one_batch():
    torch.manual_seed(0)
    m = Vishalam(TINY)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 300, (4, 32))
    for _ in range(150):
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_grad_checkpointing_same_grads():
    torch.manual_seed(0)
    m = Vishalam(TINY)
    x = torch.randint(0, 300, (2, 16))
    _, l1 = m(x, x)
    l1.backward()
    g1 = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    m.grad_checkpointing = True
    _, l2 = m(x, x)
    l2.backward()
    for a, p in zip(g1, m.parameters()):
        assert torch.allclose(a, p.grad, atol=1e-6)


def test_chunked_loss_matches_full_loss_and_grads(monkeypatch):
    import vishalam.model.model as mm

    monkeypatch.setattr(mm, "LOSS_CHUNK", 7)  # several uneven chunks
    torch.manual_seed(0)
    m = Vishalam(TINY)
    x = torch.randint(0, 300, (2, 16))
    y = torch.randint(0, 300, (2, 16))
    mask = (torch.rand(2, 16) > 0.3).float()
    _, full = m(x, y, loss_mask=mask, z_loss=1e-3, return_logits=True)
    full.backward()
    g_full = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    logits, chunked = m(x, y, loss_mask=mask, z_loss=1e-3)
    assert logits is None
    chunked.backward()
    assert torch.allclose(full, chunked, atol=1e-5)
    for a, p in zip(g_full, m.parameters()):
        assert torch.allclose(a, p.grad, atol=1e-5)
