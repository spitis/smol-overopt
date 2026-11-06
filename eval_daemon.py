"""Label eval generations with the quality judge as they appear and write quality curves to RUN/quality_curve.jsonl.
usage: python eval_daemon.py SRC RUN [--second MODEL]
  SRC  the training run dir: a local path, or host:dir (pulled with rsync, which must be installed)
  RUN  local dir where eval files, labels and quality_curve.jsonl go (may equal SRC for a local run)
Each eval file step_XXXXX.jsonl (complete files only; grpo.py writes them atomically) gets step_XXXXX.lab.jsonl with h_judge, c_judge, raw_judge.
Answers whose label is missing are retried up to --label_retries rounds; a step is recorded once every answer has a label or the retries are used up,
and its curve row reports n (answers) and n_labelled. Curve rows carry the run id from args.json; rows of other runs in the same file are ignored.
--second MODEL optionally labels the first --n_second answers with another judge model (h_judge2, ...), used by tools/early_stop.py as referee.
The second model must be served by the same endpoint. Steps that are already recorded are backfilled when --second is given later.
Judge config: environment variables, see judge.py."""
import os, json, time, subprocess, argparse
import numpy as np
import judge
from curve import load_curve, run_id

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("run"); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--second", default=None); ap.add_argument("--n_second", type=int, default=150); ap.add_argument("--hours", type=float, default=6.0); ap.add_argument("--poll", type=int, default=120)
ap.add_argument("--label_retries", type=int, default=3, help="rounds of relabelling answers whose label is missing before a step is recorded as is")
ap.add_argument("--once", action="store_true", help="label what exists and exit")
a = ap.parse_args(); os.makedirs(a.run + "/eval", exist_ok=True); t0 = time.time(); curve = a.run + "/quality_curve.jsonl"
same = os.path.realpath(a.src) == os.path.realpath(a.run)
mean = lambda x: float(np.mean(x)) if x else float("nan")
q = lambda h, c: (h + c) / 2
txt = lambda r: r["text"] or "(empty answer)"


def fill(R, suf, model, limit, rounds, **kw):
    """Label rows R[:limit] lacking h_<suf>; up to `rounds` passes. Returns the number of labelled rows."""
    for _ in range(rounds):
        idx = [i for i in range(min(limit, len(R))) if R[i].get("h_" + suf) is None]
        if not idx: break
        res = judge.score_many([(R[i]["prompt"], txt(R[i])) for i in idx], model, a.workers, raw=True, **kw)
        for i, (h, c, t) in zip(idx, res):
            if h is not None or "raw_" + suf not in R[i]: R[i]["h_" + suf], R[i]["c_" + suf], R[i]["raw_" + suf] = h, c, t
    return sum(r.get("h_" + suf) is not None for r in R[:limit])


def make_row(step, R, rid):
    ok = [r for r in R if r["h_judge"] is not None]; okf = [r for r in ok if r["finished"]]
    okg = [r for r in R[:a.n_second] if r.get("h_judge2") is not None]
    return dict(step=step, run_id=rid, n=len(R), n_labelled=len(ok), quality_h=mean([r["h_judge"] for r in ok]), quality_c=mean([r["c_judge"] for r in ok]),
                quality=mean([q(r["h_judge"], r["c_judge"]) for r in ok]), quality_finished=mean([q(r["h_judge"], r["c_judge"]) for r in okf]),
                quality2=mean([q(r["h_judge2"], r["c_judge2"]) for r in okg]) if okg else None, quality2_n=len(okg), second=a.second if okg else None,
                proxy=mean([r["proxy"] for r in R]), len=mean([r["n_tok"] for r in R]), empty=mean([not r["text"] for r in R]), fin=mean([r["finished"] for r in R]))


def load_labelled(step, rid):
    """Eval rows for a step, reusing labels of the same run from an earlier pass."""
    R = [json.loads(l) for l in open(f"{a.run}/eval/step_{step:05d}.jsonl")]
    p = f"{a.run}/eval/step_{step:05d}.lab.jsonl"
    if os.path.exists(p):
        old = [json.loads(l) for l in open(p)]
        if len(old) == len(R) and all(o.get("run_id") == rid and o["prompt"] == r["prompt"] and o["text"] == r["text"] for o, r in zip(old, R)): R = old
    for r in R: r["run_id"] = rid
    return R


def save(step, R):
    with open(f"{a.run}/eval/step_{step:05d}.lab.jsonl", "w") as o:
        for r in R: o.write(json.dumps(r) + "\n")


while True:
    if not same:
        src = a.src.rstrip("/")
        subprocess.run(["rsync", "-aq", "--delete", "--exclude=*.tmp", "--exclude=*.lab.jsonl", f"{src}/eval/", a.run + "/eval/"])
        subprocess.run(["rsync", "-aq", f"{src}/log.jsonl", f"{src}/args.json", a.run + "/"])
    rid = run_id(a.run); rows = {r["step"]: r for r in load_curve(a.run)}
    files = sorted(f for f in os.listdir(a.run + "/eval") if f.startswith("step_") and f.endswith(".jsonl") and ".lab" not in f and ".tmp" not in f)
    for f in files:
        s = int(f[5:10]); row = rows.get(s)
        if row is not None and not (a.second and row.get("second") != a.second): continue
        R = load_labelled(s, rid)
        fill(R, "judge", None, len(R), a.label_retries)
        if not any(r["h_judge"] is not None for r in R): print("no valid judge labels for", f, "(check the endpoint and key); will retry", flush=True); continue
        if a.second: fill(R, "judge2", a.second, a.n_second, a.label_retries, retries=2, timeout=30)
        save(s, R); row = make_row(s, R, rid)
        with open(curve, "a") as o: o.write(json.dumps(row) + "\n")
        print(row, flush=True)
    if a.once or time.time() - t0 > a.hours * 3600: break
    time.sleep(a.poll)
