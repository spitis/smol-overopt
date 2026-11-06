"""Label a jsonl of {prompt, text, ...} rows with the quality judge. Adds h_judge, c_judge (0-4, null on failure). usage: python label.py in.jsonl out.jsonl"""
import sys, json, argparse
import judge
ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.inp)]
sc = judge.score_many([(r["prompt"], r["text"] if r["text"] else "(empty answer)") for r in rows], workers=a.workers, raw=True)
for r, (h, c, t) in zip(rows, sc): r["h_judge"], r["c_judge"], r["raw_judge"] = h, c, t
print("failures", sum(h is None for h, c, _ in sc), "of", len(sc), file=sys.stderr)
with open(a.out, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
