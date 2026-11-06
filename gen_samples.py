"""Sample starting-policy answers on train prompts (the data the reward model is trained on). usage: python gen_samples.py --out data/base.jsonl"""
import json, argparse, time
from common import *
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct"); ap.add_argument("--prompts", default="data/prompts.json")
ap.add_argument("--n_prompts", type=int, default=3000, help="train prompts used (3000 x k=2 = 6000 samples, as in the reference RM)")
ap.add_argument("--k", type=int, default=2, help="samples per prompt"); ap.add_argument("--max_new", type=int, default=256); ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--offset", type=int, default=0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--out", required=True)
a = ap.parse_args(); torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)
tr, _ = load_prompts(a.prompts); tr = filter_prompts(tok, tr, "train prompts"); P = tr[a.offset:a.offset + a.n_prompts]
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
pr = [p for p in P for _ in range(a.k)]
t = time.time(); o = generate(m, tok, pr, a.max_new, a.temp, a.bs)
print("gen time", time.time() - t, "n", len(o), "mean len", sum(len(x["comp_ids"]) for x in o) / len(o), "fin", sum(x["finished"] for x in o) / len(o))
with open(a.out, "w") as f:
    for p, x in zip(pr, o): f.write(json.dumps(dict(prompt=p, text=x["text"], n_tok=len(x["comp_ids"]), finished=x["finished"], model=a.model)) + "\n")
