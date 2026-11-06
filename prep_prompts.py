"""HelpSteer2 -> data/prompts.json (deterministic train/eval prompt split).
Single-turn prompts only (no multi-turn '<extra_id' markers), more than 20 and at most 1200 characters, unique; sorted, shuffled with
random.Random(0); the first 300 are the held-out eval prompts and the rest (6489) are train prompts.
Downloads nvidia/HelpSteer2 train and validation from the Hugging Face Hub unless local .jsonl.gz paths are given with --files."""
import argparse, gzip, json, os, random

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/prompts.json"); ap.add_argument("--n_eval", type=int, default=300)
ap.add_argument("--files", nargs="*", help="local HelpSteer2 train/validation .jsonl.gz files, in this order (default: download)")
a = ap.parse_args()
files = a.files
if not files:
    from huggingface_hub import hf_hub_download
    files = [hf_hub_download("nvidia/HelpSteer2", f, repo_type="dataset") for f in ("train.jsonl.gz", "validation.jsonl.gz")]
seen = set()
for f in files:
    for line in gzip.open(f, "rt"):
        p = json.loads(line)["prompt"]
        if "<extra_id" not in p and 20 < len(p) <= 1200: seen.add(p)
prompts = sorted(seen); random.Random(0).shuffle(prompts)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(dict(train=prompts[a.n_eval:], eval=prompts[:a.n_eval]), open(a.out, "w"))
print(f"{len(prompts)} unique prompts: {len(prompts) - a.n_eval} train, {a.n_eval} eval -> {a.out}")
