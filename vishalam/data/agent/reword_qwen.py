"""Reword a fraction of training requests with Qwen2.5-7B-Instruct (Apache-2.0) for more natural,
varied English. Runs on a Kaggle T4 with vLLM + the official AWQ 4-bit build.

    python -m vishalam.data.agent.reword_qwen --inp data/m1/train.jsonl --out data/m1/train_mixed.jsonl --fraction 0.4

Safety rails (a rewrite is discarded and the template request kept if any fails):
  * every `backticked` item (function, parameter and file names, literals) survives verbatim
  * every number in the original survives
  * length stays within 0.5x - 2x, no code blocks, no new backticked names
Reworded examples are tagged request_source = "ai-reworded" so the README can disclose the share.
Hand-written *test* requests are never touched; hand-written *seed* requests are used as style
examples in the prompt.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
_TICKS = re.compile(r"`[^`]+`")
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")

SYSTEM = (
    "You rewrite programming task requests so they sound like different real people wrote them. "
    "Keep the exact meaning and every requirement. Keep every piece of text that is inside backticks "
    "exactly as it is, including the backticks. Keep every number. Do not add requirements, examples, "
    "code, or explanations. Output only the rewritten request."
)


def valid(orig: str, new: str) -> bool:
    new = new.strip()
    if not new or "```" in new or len(new) < 0.5 * len(orig) or len(new) > 2 * len(orig) + 40:
        return False
    o_ticks, n_ticks = _TICKS.findall(orig), _TICKS.findall(new)
    if any(t not in new for t in o_ticks) or not set(n_ticks) <= set(o_ticks):
        return False
    o_nums = [n for n in _NUM.findall(_TICKS.sub("", orig))]
    return all(n in new for n in o_nums)


def build_prompt(tokenizer, request: str, seeds: list[str], rng: random.Random) -> str:
    style = ""
    if seeds:
        shots = rng.sample(seeds, min(3, len(seeds)))
        style = "Examples of how real users phrase requests:\n" + "\n".join(f"- {s}" for s in shots) + "\n\n"
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{style}Rewrite this request:\n{request}"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fraction", type=float, default=0.4)
    ap.add_argument("--seeds", default="handwritten/requests.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel GPUs")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rng = random.Random(args.seed)
    eps = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    seeds = []
    if Path(args.seeds).exists():
        seeds = [json.loads(l)["request"] for l in open(args.seeds, encoding="utf-8")
                 if l.strip() and json.loads(l).get("split") == "seed"]
    pick = [i for i in range(len(eps)) if rng.random() < args.fraction]

    llm = LLM(MODEL, quantization="awq", dtype="float16", max_model_len=2048, tensor_parallel_size=args.tp,
              gpu_memory_utilization=0.9)
    tok = llm.get_tokenizer()
    prompts = [build_prompt(tok, eps[i]["request"], seeds, rng) for i in pick]
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=300, seed=args.seed)
    outs = llm.generate(prompts, params)

    kept = 0
    for i, o in zip(pick, outs):
        new = o.outputs[0].text.strip().strip('"')
        if valid(eps[i]["request"], new):
            eps[i]["template_request"] = eps[i]["request"]
            eps[i]["request"] = new
            eps[i]["request_source"] = "ai-reworded"
            kept += 1
    with open(args.out, "w", encoding="utf-8") as f:
        for ep in eps:
            f.write(json.dumps(ep) + "\n")
    print(json.dumps({"episodes": len(eps), "attempted": len(pick), "reworded": kept,
                      "ai_reworded_share": round(kept / max(1, len(eps)), 4)}))


if __name__ == "__main__":
    main()
