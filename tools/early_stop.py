"""Offline early stopping with a stopping judge. usage: python tools/early_stop.py RUN_DIR [RUN_DIR ...]  (EVERY=25 env var sets the eval spacing)
Stop rule: stop at the first eval whose referee score fails to beat the previous best (patience 1), or fails twice in a row (patience 2).
Regret = best judged quality over the evals minus judged quality at the stop step. Steps are sorted and de-duplicated. Referees: the training reward (always), and a second judge
model on the first 150 prompts if the run was labelled with eval_daemon.py --second (then quality is the primary judge on all prompts)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from curve import load_curve

L = lambda p: [json.loads(l) for l in open(p)]


def sc(R, m, n=None):
    R = R[:n] if n else R; v = [(r["h_" + m] + r["c_" + m]) / 2 for r in R if r.get("h_" + m) is not None]; return float(np.mean(v)) if v else np.nan


def stop_idx(s, patience=1):
    best = -1e9; bi = 0; bad = 0
    for i, x in enumerate(s):
        if x != x: continue
        if x > best: best, bi, bad = x, i, 0
        else:
            bad += 1
            if bad >= patience: return i
    return len(s) - 1


def table(d, every=int(os.environ.get("EVERY", 25))):
    st = [g["step"] for g in load_curve(d) if g["step"] % every == 0 and os.path.exists(f"{d}/eval/step_{g['step']:05d}.lab.jsonl")]
    Rs = {s: L(f"{d}/eval/step_{s:05d}.lab.jsonl") for s in st}
    series = dict(quality=[sc(Rs[s], "judge") for s in st], proxy=[float(np.mean([r["proxy"] for r in Rs[s]])) for s in st])
    refs = [("reward", "quality", "proxy")]
    if any(r.get("h_judge2") is not None for s in st for r in Rs[s][:150]):   # steps without second labels give NaN and are skipped
        series["quality150"] = [sc(Rs[s], "judge", 150) for s in st]; series["judge2"] = [sc(Rs[s], "judge2", 150) for s in st]
        refs += [("judge2 on 150", "quality", "judge2")]
    out = {}
    for name, gold, ref in refs:
        for pat in (1, 2):
            g = np.array(series[gold]); i = stop_idx(series[ref], pat); out[(name, pat)] = (st[i], np.nanmax(g) - g[i], st[int(np.nanargmax(g))], np.nanmax(g))
    return out


if __name__ == "__main__":
    print("| run | referee | patience | stop step | regret | best-quality step | best quality |\n|---|---|---|---|---|---|---|")
    for run in sys.argv[1:]:
        for (n, p), (s, r, bs, bg) in table(run.rstrip("/")).items(): print(f"| {os.path.basename(run.rstrip('/'))} | {n} | {p} | {s} | {r:.3f} | {bs} | {bg:.2f} |")
