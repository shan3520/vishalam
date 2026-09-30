"""The Vishalam agent protocol: how an episode is laid out as tokens.

    <|request|>Write a function `second_largest` ...<|end|>
    <|write_file|><|path|>test_second_largest.py<|content|>...<|end|>      <- model
    <|result|>wrote test_second_largest.py (9 lines)<|end|>                <- runtime
    <|write_file|><|path|>second_largest.py<|content|>...<|end|>           <- model
    <|result|>...<|end|>
    <|run_tests|><|end|>                                                   <- model
    <|result|>F. 1 failed, 1 passed ...<|end|>                             <- runtime
    <|edit_file|><|path|>second_largest.py<|old|>...<|new|>...<|end|>      <- model
    ...
    <|done|><|message|>Implemented ...<|end|><|endoftext|>                 <- model

Only model-produced tokens get training loss. Request text, file contents and tool results are
encoded with special tokens disabled, so nothing inside them can forge protocol tokens.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from vishalam.tokenizer.bpe import Tokenizer

# tool name -> ordered argument names
TOOLS: dict[str, list[str]] = {
    "list_files": [],
    "read_file": ["path"],
    "write_file": ["path", "content"],
    "edit_file": ["path", "old", "new"],
    "search": ["pattern"],
    "run_tests": [],
    "run_python": ["code"],
    "done": ["message"],
}

MAX_RESULT_CHARS = 1500


@dataclass
class ToolCall:
    name: str
    args: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Step:
    call: ToolCall
    result: str | None  # None for `done`

    def to_dict(self) -> dict:
        return {"call": self.call.to_dict(), "result": self.result}

    @staticmethod
    def from_dict(d: dict) -> "Step":
        return Step(ToolCall(d["call"]["name"], d["call"]["args"]), d["result"])


def truncate_result(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return text[:head] + f"\n... [{len(text) - limit} chars truncated] ...\n" + text[-tail:]


def _call_ids(tok: Tokenizer, call: ToolCall) -> list[int]:
    ids = [tok.sid(f"<|{call.name}|>")]
    for arg in TOOLS[call.name]:
        ids.append(tok.sid(f"<|{arg}|>"))
        ids.extend(tok.encode_ordinary(call.args.get(arg, "")))
    ids.append(tok.sid("<|end|>"))
    return ids


def request_ids(tok: Tokenizer, request: str) -> list[int]:
    return [tok.sid("<|request|>")] + tok.encode_ordinary(request) + [tok.sid("<|end|>")]


def result_ids(tok: Tokenizer, result: str) -> list[int]:
    return [tok.sid("<|result|>")] + tok.encode_ordinary(truncate_result(result)) + [tok.sid("<|end|>")]


def encode_episode(tok: Tokenizer, request: str, steps: list[Step], finished: bool = True
                   ) -> tuple[list[int], list[int]]:
    """Returns (ids, is_model) where is_model[i] = 1 if token i is produced by the model."""
    ids = request_ids(tok, request)
    is_model = [0] * len(ids)
    for st in steps:
        c = _call_ids(tok, st.call)
        ids += c
        is_model += [1] * len(c)
        if st.result is not None:
            r = result_ids(tok, st.result)
            ids += r
            is_model += [0] * len(r)
    if finished:
        ids.append(tok.eot_id)
        is_model.append(1)
    return ids, is_model


def training_example(tok: Tokenizer, request: str, steps: list[Step]) -> tuple[list[int], list[int], list[int]]:
    """(inputs, targets, loss_mask) for next-token training; loss only where the target is model-made."""
    ids, is_model = encode_episode(tok, request, steps)
    return ids[:-1], ids[1:], is_model[1:]


class ParseError(ValueError):
    pass


def parse_call(tok: Tokenizer, ids: list[int]) -> ToolCall:
    """Parse model-generated tokens for one call (tool token ... <|end|>) into a ToolCall."""
    if not ids:
        raise ParseError("empty call")
    first = tok.id_to_special.get(ids[0])
    name = first[2:-2] if first else None
    if name not in TOOLS:
        raise ParseError(f"call does not start with a tool token: {tok.decode(ids[:5])!r}")
    expected = TOOLS[name]
    args: dict[str, str] = {}
    cur: str | None = None
    buf: list[int] = []
    seen: list[str] = []
    for t in ids[1:]:
        sp = tok.id_to_special.get(t)
        if sp is None:
            if cur is None:
                raise ParseError(f"text before first argument of {name}")
            buf.append(t)
            continue
        key = sp[2:-2]
        if cur is not None:
            args[cur] = tok.decode(buf)
        if sp == "<|end|>":
            cur = None
            break
        if key not in expected:
            raise ParseError(f"unexpected token {sp} in {name}")
        cur, buf = key, []
        seen.append(key)
    else:
        raise ParseError(f"{name} call not terminated with <|end|>")
    if seen != expected:
        raise ParseError(f"{name} expects arguments {expected}, got {seen}")
    return ToolCall(name, args)
