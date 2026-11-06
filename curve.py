"""Shared reader for RUN/quality_curve.jsonl: rows of the current run only (matching the run id in args.json), one row per step (the last
one written wins), sorted by step."""
import json, os


def run_id(run):
    try: return json.load(open(run + "/args.json")).get("run_id")
    except (OSError, ValueError): return None


def load_curve(run):
    p = run + "/quality_curve.jsonl"
    if not os.path.exists(p): return []
    rid = run_id(run); by = {}
    for l in open(p):
        if not l.strip(): continue
        r = json.loads(l)
        if rid is not None and r.get("run_id") != rid: continue
        by[r["step"]] = r
    return [by[s] for s in sorted(by)]
