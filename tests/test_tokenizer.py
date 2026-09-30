import random

import pytest

from vishalam.tokenizer.bpe import NUM_SPECIAL_SLOTS, Tokenizer, special_token_list
from vishalam.tokenizer.bpe_train import count_chunks, train_bpe

CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "def second_largest(nums):\n    s = sorted(set(nums))\n    return s[-2]\n",
    "class Stack:\n    def __init__(self):\n        self.items = []\n",
    "The quick brown fox jumps over the lazy dog. Numbers: 12345 and 3.14!\n",
    "for i in range(10):\n        if i % 2 == 0:\n            print(i)\n",
] * 50


@pytest.fixture(scope="module")
def tok():
    counts = count_chunks(CORPUS)
    merges = train_bpe(counts, vocab_size=256 + NUM_SPECIAL_SLOTS + 300, min_frequency=1, verbose=False)
    return Tokenizer(merges)


def test_merges_compress(tok):
    text = CORPUS[1]
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 2


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello world",
        "def f(x):\n\treturn x  # tabs and    spaces\n",
        "ünïcödé ✓ 日本語 🙂 emoji",
        "\x00\x01 control bytes \r\n windows newlines",
    ],
)
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_random_bytes(tok):
    rng = random.Random(0)
    for _ in range(50):
        s = "".join(chr(rng.randint(32, 0x2FFF)) for _ in range(rng.randint(0, 80)))
        assert tok.decode(tok.encode(s)) == s


def test_special_tokens_only_when_allowed(tok):
    text = "<|write_file|><|path|>a.py<|end|>"
    ids_special = tok.encode(text, allow_special=True)
    assert ids_special[0] == tok.sid("<|write_file|>")
    assert tok.sid("<|end|>") in ids_special
    ids_plain = tok.encode(text)  # e.g. tool output echoing the string: must NOT become specials
    assert not any(i in tok.id_to_special for i in ids_plain)
    assert tok.decode(ids_special) == text == tok.decode(ids_plain)


def test_vocab_layout(tok):
    assert tok.vocab_size == 256 + len(tok.merges) + NUM_SPECIAL_SLOTS
    assert len(special_token_list()) == NUM_SPECIAL_SLOTS
    assert tok.eot_id == 256 + len(tok.merges)


def test_save_load(tok, tmp_path):
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = Tokenizer.load(p)
    s = CORPUS[2] + "<|done|>"
    assert tok2.encode(s, allow_special=True) == tok.encode(s, allow_special=True)


def test_training_deterministic():
    counts = count_chunks(CORPUS)
    m1 = train_bpe(counts, 256 + NUM_SPECIAL_SLOTS + 100, 1, verbose=False)
    m2 = train_bpe(counts, 256 + NUM_SPECIAL_SLOTS + 100, 1, verbose=False)
    assert m1 == m2
