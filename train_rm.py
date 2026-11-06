"""Train the scalar reward model (SmolLM2-360M-Instruct trunk + linear head) by regression on judge quality (mean of h and c).
usage: python train_rm.py --data data/base.lab.jsonl --out data/rm360.pt   (holds out 10% of prompts for Spearman/Pearson)"""
import sys, json, argparse, random, time
import torch, numpy as np
from proxies import RM

ap = argparse.ArgumentParser(); ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--max_rows", type=int, default=0); ap.add_argument("--label", default="judge"); ap.add_argument("--hold_frac", type=float, default=0.1)
a = ap.parse_args(); random.seed(0); torch.manual_seed(0)


def rank(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    for v in np.unique(x): r[x == v] = r[x == v].mean()   # average ranks for ties
    return r


R = []
for f in a.data: R += [json.loads(l) for l in open(f)]
R = [r for r in R if r.get(f"h_{a.label}") is not None]
for r in R: r["y"] = (r[f"h_{a.label}"] + r[f"c_{a.label}"]) / 2
prompts = sorted({r["prompt"] for r in R}); random.shuffle(prompts); hp = set(prompts[:int(len(prompts) * a.hold_frac)])
tr = [r for r in R if r["prompt"] not in hp]; ho = [r for r in R if r["prompt"] in hp]
if a.max_rows: tr = tr[:a.max_rows]
if len({r["y"] for r in tr}) < 2: sys.exit("train_rm.py: the training labels are constant (or there are no training rows); a reward model cannot be fitted. Check that the judge returned varied scores.")
print("train", len(tr), "hold", len(ho), flush=True)
rm = RM(a.model).cuda(); rm.trunk.gradient_checkpointing_enable(); rm.train()
ys = np.array([r["y"] for r in tr]); rm.mu, rm.sd = float(ys.mean()), float(ys.std())
opt = torch.optim.AdamW([{"params": rm.trunk.parameters(), "lr": a.lr}, {"params": rm.head.parameters(), "lr": 1e-3}], weight_decay=0.0)
sc = torch.amp.GradScaler(); nsteps = a.epochs * (len(tr) // a.bs); step = 0; t0 = time.time()


def evaluate():
    rm.eval(); s = np.array(rm.score([r["prompt"] for r in ho], [r["text"] for r in ho])); rm.train(); y = np.array([r["y"] for r in ho])
    return float(np.corrcoef(rank(s), rank(y))[0, 1]), float(np.corrcoef(s, y)[0, 1])


for ep in range(a.epochs):
    random.shuffle(tr)
    for b in range(0, len(tr) - a.bs + 1, a.bs):
        B = tr[b:b + a.bs]; I, M = rm.encode([r["prompt"] for r in B], [r["text"] for r in B])
        y = torch.tensor([(r["y"] - rm.mu) / rm.sd for r in B]).cuda()
        with torch.autocast("cuda", dtype=torch.float16): p = rm(I, M)
        loss = ((p.float() - y) ** 2).mean()
        opt.zero_grad(); sc.scale(loss).backward(); sc.unscale_(opt); torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0); sc.step(opt); sc.update(); step += 1
        if step % 50 == 0: print(f"step {step}/{nsteps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
        if step % 300 == 0: print("hold spearman/pearson", evaluate(), flush=True)
sp, pe = evaluate(); print("FINAL hold spearman", sp, "pearson", pe, flush=True)
torch.save({"trunk": rm.trunk.state_dict(), "head": rm.head.state_dict(), "mu": rm.mu, "sd": rm.sd, "name": a.model, "spearman": sp, "pearson": pe}, a.out)
