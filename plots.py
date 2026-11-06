"""Plot judged quality and learned reward for one or more runs. usage: python plots.py RUN_DIR [RUN_DIR ...] [--out curves.png]
Per run (RUN_DIR needs log.jsonl and quality_curve.jsonl): quality (solid) and reward (dashed) vs step, and both vs sqrt(KL) to the start policy
(KL = sequence-level sample estimate, averaged over the logged steps among the 4 up to each eval; the final step uses one extra batch of samples).
Steps are sorted and de-duplicated; steps without a value for the plotted key are skipped."""
import json, os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curve import load_curve
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--out", default="curves.png"); ap.add_argument("--quality_key", default="quality")
a = ap.parse_args()
L = lambda p: [json.loads(l) for l in open(p)] if os.path.exists(p) else []
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
for k, run in enumerate(a.runs):
    log = L(run + "/log.jsonl"); qc = [g for g in load_curve(run) if g.get(a.quality_key) is not None]; name = os.path.basename(run.rstrip("/")); c = f"C{k}"
    kl = {r["step"]: r["kl"] for r in log if "kl" in r}
    st = [g["step"] for g in qc]; qu = [g[a.quality_key] for g in qc]; px = [g["proxy"] for g in qc]
    def sqkl(s):
        v = [kl[s - j] for j in range(4) if s - j in kl]
        return float(np.sqrt(max(np.mean(v), 0))) if v else np.nan
    sk = [sqkl(s) for s in st]
    ax[0].plot(st, qu, "o-", color=c, label=name); ax[1].plot(sk, qu, "o-", color=c, label=name); ax[2].plot(sk, px, "^--", color=c, label=name)
    if qu: i = int(np.argmax(qu)); print(f"{name}: quality {qu[0]:.2f} -> peak {qu[i]:.2f} at step {st[i]} (sqrt KL {sk[i]:.1f}); last {qu[-1]:.2f} at step {st[-1]}; reward {px[0]:.2f} -> {px[-1]:.2f}")
for x, (xl, yl) in zip(ax, [("step", "judged quality (0-4)"), ("sqrt KL", "judged quality (0-4)"), ("sqrt KL", "learned reward (judge units)")]): x.set_xlabel(xl); x.set_ylabel(yl)
ax[0].set_title("quality vs step"); ax[1].set_title("quality vs sqrt(KL)"); ax[2].set_title("reward vs sqrt(KL)"); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(a.out, dpi=100); print("wrote", a.out)
