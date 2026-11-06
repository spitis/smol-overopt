#!/bin/sh
# Smoke test: 5 GRPO steps with a dummy reward on a tiny prompt file. No judge, no reward model, no HelpSteer2 download needed.
# Needs one CUDA GPU (grpo.py uses .cuda()) and access to the SmolLM2-135M-Instruct weights. usage: sh tests/smoke.sh
set -e
cd "$(dirname "$0")/.."
T=$(mktemp -d)
python - "$T/prompts.json" <<'PY'
import json, sys
topics = ["the water cycle", "binary search", "photosynthesis", "the French revolution", "compound interest", "sorting algorithms", "black holes", "the immune system"]
mk = lambda i: f"Explain {topics[i % len(topics)]} in a few sentences (variant {i})."
json.dump(dict(train=[mk(i) for i in range(64)], eval=[mk(i) for i in range(100, 108)]), open(sys.argv[1], "w"))
PY
python grpo.py --proxy dummy --prompts "$T/prompts.json" --steps 5 --B 4 --G 4 --max_new 48 --eval_every 5 --eval_n 8 --ckpt_every 1000 --out "$T/run"
python - "$T/run" <<'PY'
import json, sys, os
d = sys.argv[1]
log = [json.loads(l) for l in open(d + "/log.jsonl")]
train = [r for r in log if "proxy" in r and "kl" in r]
assert len(train) == 5, len(train)
assert all(r["kl"] == r["kl"] for r in train), "nan kl"
assert any("kl" in r and "proxy" not in r for r in log), "no final-step kl"
assert os.path.exists(d + "/ckpt/step_5.pt"), "no final checkpoint"
assert sorted(f for f in os.listdir(d + "/eval") if f.endswith(".jsonl")) == ["step_00000.jsonl", "step_00005.jsonl"]
print("smoke test passed:", [round(r["proxy"], 3) for r in train])
PY
! python grpo.py --proxy dummy --prompts "$T/prompts.json" --steps 1 --temp 0.7 --out "$T/x" 2>/dev/null
! python grpo.py --proxy dummy --prompts "$T/prompts.json" --steps 1 --out "$T/run" 2>/dev/null
echo "argument checks passed"
rm -rf "$T"
