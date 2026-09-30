"""The agent loop: model proposes a tool call -> runtime executes it -> result goes back in context."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from vishalam.agent.format import ParseError, Step, ToolCall, parse_call, request_ids, result_ids
from vishalam.agent.tools import Workspace
from vishalam.infer.generate import Session
from vishalam.tokenizer.bpe import Tokenizer


@dataclass
class AgentResult:
    steps: list[Step] = field(default_factory=list)
    finished: bool = False          # model called `done`
    stop_reason: str = ""
    test_runs: int = 0
    parse_errors: int = 0
    message: str = ""


class Agent:
    def __init__(self, model, cfg, device: str, tok: Tokenizer, max_steps: int = 24,
                 max_test_runs: int = 4, temperature: float = 0.0, seed: int | None = None,
                 max_call_tokens: int = 1500):
        self.model, self.cfg, self.device, self.tok = model, cfg, device, tok
        self.max_steps = max_steps
        self.max_test_runs = max_test_runs  # 1 first run + up to 3 fix attempts
        self.temperature = temperature
        self.max_call_tokens = max_call_tokens
        self.gen = None
        if seed is not None:
            self.gen = torch.Generator(device=device if device != "cpu" else "cpu").manual_seed(seed)
        self.stop_ids = {tok.sid("<|end|>"), tok.eot_id}

    def run(self, request: str, ws: Workspace, on_event: Callable[[dict], None] | None = None) -> AgentResult:
        emit = on_event or (lambda e: None)
        res = AgentResult()
        sess = Session(self.model, self.cfg, self.device)
        try:
            sess.feed(request_ids(self.tok, request))
        except OverflowError:
            res.stop_reason = "request too long"
            return res
        for _ in range(self.max_steps):
            emit({"type": "thinking"})
            out = sess.sample(self.stop_ids, self.max_call_tokens, self.temperature, generator=self.gen)
            if not out or out[-1] == self.tok.eot_id and len(out) == 1:
                res.stop_reason = "model stopped without calling done"
                break
            try:
                call = parse_call(self.tok, out)
            except ParseError as e:
                res.parse_errors += 1
                err = f"error: invalid tool call ({e})"
                emit({"type": "error", "text": err})
                if res.parse_errors >= 3 or sess.remaining < 64:
                    res.stop_reason = "too many invalid tool calls"
                    break
                sess.feed(result_ids(self.tok, err))
                continue
            emit({"type": "call", "name": call.name, "args": call.args})
            if call.name == "done":
                res.steps.append(Step(call, None))
                res.finished, res.message, res.stop_reason = True, call.args.get("message", ""), "done"
                emit({"type": "done", "message": res.message})
                break
            if call.name == "run_tests":
                res.test_runs += 1
            result = ws.execute(call)
            res.steps.append(Step(call, result))
            emit({"type": "result", "name": call.name, "text": result})
            if call.name == "run_tests" and res.test_runs >= self.max_test_runs and "failed" in result:
                res.stop_reason = f"tests still failing after {res.test_runs} runs"
                break
            try:
                sess.feed(result_ids(self.tok, result))
            except OverflowError:
                res.stop_reason = "context window full"
                break
        else:
            res.stop_reason = "step limit reached"
        return res
