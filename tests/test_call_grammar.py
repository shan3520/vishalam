"""Constrained decoding: even an untrained model can only emit tool calls that parse."""
import torch

from vishalam.agent.format import TOOLS, CallGrammar, parse_call
from vishalam.infer.generate import Session
from vishalam.model.config import ModelConfig
from vishalam.model.model import Vishalam
from vishalam.tokenizer.bpe import NUM_SPECIAL_SLOTS, Tokenizer
from vishalam.tokenizer.bpe_train import count_chunks, train_bpe


def tiny_tok():
    texts = ["def add(a, b):\n    return a + b\n", "Write a function that adds two numbers."] * 20
    return Tokenizer(train_bpe(count_chunks(texts), 256 + NUM_SPECIAL_SLOTS + 64, verbose=False))


def test_allowed_follows_the_protocol():
    tok = tiny_tok()
    g = CallGrammar(tok)
    first, text = g.allowed([])
    assert not text and {tok.id_to_special[t] for t in first} == {f"<|{n}|>" for n in TOOLS}
    wf, path, content, end = (tok.sid(s) for s in ("<|write_file|>", "<|path|>", "<|content|>", "<|end|>"))
    assert g.allowed([wf]) == ((path,), False)                       # argument token forced
    some = tok.encode_ordinary("a.py")
    assert g.allowed([wf, path] + some) == ((content,), True)         # text, or the next argument
    assert g.allowed([wf, path] + some + [content]) == ((end,), True)  # last argument: text or <|end|>
    assert g.allowed([tok.sid("<|run_tests|>")]) == ((end,), False)  # no arguments


def test_untrained_model_only_emits_parseable_calls():
    tok = tiny_tok()
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_heads=4, ffn_dim=64, max_seq_len=512)
    model = Vishalam(cfg).eval()
    g, stop = CallGrammar(tok), {tok.sid("<|end|>"), tok.eot_id}
    gen = torch.Generator().manual_seed(1)
    names = set()
    for _ in range(12):
        s = Session(model, cfg, "cpu")
        s.feed([tok.sid("<|request|>")] + tok.encode_ordinary("Write add.") + [tok.sid("<|end|>")])
        out = s.sample(stop, max_new=60, temperature=1.0, generator=gen, grammar=g)
        if out[-1] != tok.sid("<|end|>"):
            continue  # ran out of tokens inside an argument (only possible with max_new)
        names.add(parse_call(tok, out).name)  # never raises ParseError
    assert names  # some complete calls were produced
