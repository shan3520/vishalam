"""Vishalam local web app.

    python -m app.server                       # uses weights/vishalam-350m-agent.pt + weights/tokenizer.json
    python -m app.server --weights path/model.pt --tokenizer path/tokenizer.json
    python -m app.server --demo                # no model yet: scripted agent replays generated tasks

Binds to 127.0.0.1 only. Agent code runs in the Docker sandbox when Docker is available.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import queue
import random
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.db import History
from vishalam.agent.format import Step, ToolCall
from vishalam.agent.sandbox import DockerSandbox, SubprocessSandbox, default_sandbox
from vishalam.agent.tools import Workspace

HERE = Path(__file__).parent
WORK_ROOT = HERE / "workspaces"


class TaskIn(BaseModel):
    request: str = ""
    demo: bool = False


class ScriptedDemoAgent:
    """Replays a synthesized episode step by step, executing every tool call for real.
    Used only in --demo mode (clearly labelled in the UI): it is not the Vishalam model."""

    def __init__(self, episode: dict, delay: float = 0.6):
        self.episode, self.delay = episode, delay

    def run(self, request: str, ws: Workspace, on_event):
        for s in self.episode["steps"]:
            on_event({"type": "thinking"})
            time.sleep(self.delay)
            call = ToolCall(s["call"]["name"], s["call"]["args"])
            on_event({"type": "call", "name": call.name, "args": call.args})
            if call.name == "done":
                on_event({"type": "done", "message": call.args.get("message", "")})
                return "done"
            on_event({"type": "result", "name": call.name, "text": ws.execute(call)})
        return "done"


class App:
    def __init__(self, weights: str | None, tokenizer: str | None, demo: bool):
        self.history = History(HERE / "history.db")
        self.sandbox = default_sandbox()
        self.sandbox_name = "docker" if isinstance(self.sandbox, DockerSandbox) else "subprocess"
        self.demo = demo
        self.model_info = None
        self.agent = None
        self.lock = threading.Lock()  # one agent run at a time (CPU-bound)
        self.streams: dict[str, queue.Queue] = {}
        if weights and Path(weights).exists() and tokenizer and Path(tokenizer).exists():
            from vishalam.agent.runtime import Agent
            from vishalam.infer.generate import fastest_setup, load_model, load_tokenizer

            device, q = fastest_setup(weights)
            model, cfg, dev = load_model(weights, device, quantize_cpu=q)
            self.agent = Agent(model, cfg, dev, load_tokenizer(tokenizer))
            n = sum(p.numel() for p in model.parameters()) if not q else None
            self.model_info = {"weights": Path(weights).name, "device": dev + (" int8" if q else ""),
                               "params_m": round(n / 1e6, 1) if n else None}

    def status(self) -> dict:
        return {"model": self.model_info, "sandbox": self.sandbox_name, "demo": self.demo}

    def start(self, body: TaskIn) -> dict:
        if body.demo:
            if not self.demo and self.agent is not None:
                raise HTTPException(400, "demo tasks are only available in --demo mode")
            episode = self._demo_episode()
            request, runner, mode = episode["request"], ScriptedDemoAgent(episode), "demo"
            initial = episode.get("initial_files", {})
        else:
            if self.agent is None:
                raise HTTPException(400, "No model loaded: the trained weights are not included (see README). "
                                         "Start the app with --demo to try a demo task.")
            request = body.request.strip()
            if not request:
                raise HTTPException(400, "empty request")
            if len(request) > 4000:
                raise HTTPException(400, "request too long")
            runner, mode, initial = self.agent, "model", {}
        tid = uuid.uuid4().hex[:12]
        self.history.create(tid, request, mode)
        q: queue.Queue = queue.Queue()
        self.streams[tid] = q
        threading.Thread(target=self._run, args=(tid, request, runner, initial, q), daemon=True).start()
        return {"id": tid, "request": request, "mode": mode}

    def _demo_episode(self) -> dict:
        from vishalam.data.agent.families import TRAIN_FAMILIES, make_instance
        from vishalam.data.agent.phrasing import make_wrap
        from vishalam.data.agent.trajectories import synthesize

        rng = random.Random()
        while True:
            task = make_instance(rng.choice(TRAIN_FAMILIES), rng, "train", wrap=make_wrap("train"))
            ep = synthesize(task, rng, "fix" if rng.random() < 0.25 else "create", SubprocessSandbox(), p_bug=0.7)
            if ep:
                return ep

    def _run(self, tid: str, request: str, runner, initial: dict, q: queue.Queue) -> None:
        events: list[dict] = []

        def emit(e: dict):
            e = {**e, "t": time.time()}
            if e["type"] != "thinking":
                events.append(e)
            q.put(e)

        root = WORK_ROOT / tid
        ws = Workspace(root, self.sandbox)
        for name, content in initial.items():
            ws.write_file(name, content)
        emit({"type": "start", "request": request})
        status, reason = "error", ""
        try:
            with self.lock:
                res = runner.run(request, ws, emit)
            if isinstance(res, str):
                status, reason = ("finished", "done") if res == "done" else ("stopped", res)
            else:
                status = "finished" if res.finished else "stopped"
                reason = res.stop_reason
        except Exception as ex:  # surface errors in the UI instead of hanging
            reason = f"{type(ex).__name__}: {ex}"
            emit({"type": "error", "text": reason})
        files = ws.snapshot() if root.exists() else {}
        emit({"type": "end", "status": status, "stop_reason": reason, "files": files})
        self.history.finish(tid, status, reason, events, files)
        q.put(None)
        shutil.rmtree(root, ignore_errors=True)


def build(weights, tokenizer, demo) -> FastAPI:
    core = App(weights, tokenizer, demo)
    api = FastAPI(title="Vishalam")
    api.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @api.get("/")
    def index():
        return FileResponse(HERE / "static" / "index.html")

    @api.get("/api/status")
    def status():
        return core.status()

    @api.post("/api/tasks")
    def create(body: TaskIn):
        return core.start(body)

    @api.get("/api/tasks")
    def history():
        return core.history.list()

    @api.get("/api/tasks/{tid}")
    def task(tid: str):
        t = core.history.get(tid)
        if t is None:
            raise HTTPException(404, "unknown task")
        return t

    @api.get("/api/tasks/{tid}/events")
    async def events(tid: str):
        q = core.streams.get(tid)
        if q is None:
            raise HTTPException(404, "no live stream for this task")

        async def gen():
            while True:
                try:
                    e = q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    continue
                if e is None:
                    core.streams.pop(tid, None)
                    yield "event: close\ndata: {}\n\n"
                    return
                yield f"data: {json.dumps(e)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return api


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/vishalam-350m-agent.pt")
    ap.add_argument("--tokenizer", default="weights/tokenizer.json")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    uvicorn.run(build(args.weights, args.tokenizer, args.demo), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
