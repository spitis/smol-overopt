"""GRPO of SmolLM2-135M-Instruct against a learned reward model (default), a local generative judge, or a dummy reward.
Writes OUT/log.jsonl (per-step training stats), OUT/eval/step_XXXXX.jsonl (held-out generations with the reward, written atomically),
OUT/samples/ and OUT/ckpt/. Defaults are the standard config of the testbed (B16 G8, 256 new tokens, lr 5e-6, beta 0, 500 steps)."""
import sys, os, json, time, argparse, random, shutil, uuid
import torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from proxies import RM, GenJudge, DummyReward

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct"); ap.add_argument("--prompts", default="data/prompts.json")
ap.add_argument("--proxy", choices=["rm", "gj", "dummy"], default="rm"); ap.add_argument("--rm_ckpt", default="data/rm360.pt"); ap.add_argument("--gj_model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--steps", type=int, default=500); ap.add_argument("--B", type=int, default=16, help="prompts per step"); ap.add_argument("--G", type=int, default=8, help="samples per prompt")
ap.add_argument("--max_new", type=int, default=256); ap.add_argument("--lr", type=float, default=5e-6); ap.add_argument("--temp", type=float, default=1.0, help="sampling temperature; must be 1.0 (the policy gradient and the KL use the untempered policy, so other values would make the samples off-policy)")
ap.add_argument("--beta", type=float, default=0.0, help="per-token k3 KL penalty coefficient (0 = off)")
ap.add_argument("--eval_every", type=int, default=25); ap.add_argument("--eval_n", type=int, default=300)
ap.add_argument("--ckpt_every", type=int, default=100); ap.add_argument("--hours", type=float, default=4.0); ap.add_argument("--out", required=True)
ap.add_argument("--micro", type=int, default=4, help="sequences per backward pass"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--gen_bs", type=int, default=256)
ap.add_argument("--overwrite", action="store_true", help="clear OUT first if it already holds files (otherwise a non-empty OUT is an error)")
a = ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
if a.temp != 1.0: ap.error("--temp must be 1.0: the policy gradient and the KL estimates assume samples from the policy itself")
if os.path.isdir(a.out) and os.listdir(a.out):
    if not a.overwrite: ap.error(f"{a.out} is not empty; choose a new --out or pass --overwrite to clear it")
    for n in os.listdir(a.out):
        p = os.path.join(a.out, n); shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)
os.makedirs(a.out + "/eval", exist_ok=True); os.makedirs(a.out + "/ckpt", exist_ok=True); os.makedirs(a.out + "/samples", exist_ok=True)
json.dump(dict(vars(a), run_id=uuid.uuid4().hex), open(a.out + "/args.json", "w"))
tok = AutoTokenizer.from_pretrained(a.model)
tr, ev = load_prompts(a.prompts); tr = filter_prompts(tok, tr, "train prompts"); ev = filter_prompts(tok, ev, "eval prompts")[:a.eval_n]
master = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).cuda(); master.train()
gen = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
ref = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
if a.proxy == "rm":
    ck = torch.load(a.rm_ckpt, map_location="cpu", weights_only=False); prox = RM(ck["name"]); prox.trunk.load_state_dict(ck["trunk"]); prox.head.load_state_dict(ck["head"]); prox.mu, prox.sd = ck["mu"], ck["sd"]
    prox = prox.half().cuda().eval()
elif a.proxy == "gj":
    prox = GenJudge(a.gj_model)
else:
    prox = DummyReward()
pscore = lambda P, T: prox.score(P, T, bs=32)
opt = torch.optim.AdamW(master.parameters(), lr=a.lr, weight_decay=0.0); scaler = torch.amp.GradScaler()


def sync():
    with torch.no_grad():
        for pg, p in zip(gen.parameters(), master.parameters()): pg.data.copy_(p.data)


def batch_logp(model, seqs):
    """seqs: list of (prompt_ids, comp_ids) -> (sum of logp over completion tokens per sequence, #completion tokens, per-token logp)"""
    L = max(len(p) + len(c) for p, c in seqs); pad = tok.pad_token_id
    I = torch.full((len(seqs), L), pad); M = torch.zeros((len(seqs), L), dtype=torch.long); C = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, (p, c) in enumerate(seqs):
        n = len(p) + len(c); I[i, :n] = torch.tensor(p + c); M[i, :n] = 1; C[i, len(p):n] = True
    I, M, C = I.cuda(), M.cuda(), C.cuda()
    with torch.autocast("cuda", dtype=torch.float16): lg = model(input_ids=I, attention_mask=M).logits[:, :-1]
    Cm = C[:, 1:]; sel = lg[Cm].float(); tgt = I[:, 1:][Cm]
    lpt = -torch.nn.functional.cross_entropy(sel, tgt, reduction="none")
    rows = torch.arange(len(seqs), device=lpt.device)[:, None].expand_as(Cm)[Cm]
    out = torch.zeros(len(seqs), device=lpt.device).index_add(0, rows, lpt)
    return out, Cm.sum(-1), lpt


def do_eval(step):
    sync(); t = time.time(); o = generate(gen, tok, ev, a.max_new, a.temp, a.gen_bs); texts = [x["text"] for x in o]
    ps = pscore(ev, texts); fn, fs, fr = cross_prompt_rep(ev, texts)
    with open(f"{a.out}/eval/step_{step:05d}.jsonl.tmp", "w") as f:
        for p, x, s in zip(ev, o, ps): f.write(json.dumps(dict(prompt=p, text=x["text"], n_tok=len(x["comp_ids"]), finished=x["finished"], proxy=s)) + "\n")
    os.replace(f"{a.out}/eval/step_{step:05d}.jsonl.tmp", f"{a.out}/eval/step_{step:05d}.jsonl")
    return dict(eval_proxy=float(np.mean(ps)), eval_len=float(np.mean([len(x["comp_ids"]) for x in o])), eval_rep3=fn, eval_repsent=fs, eval_repfrac=fr,
                eval_fin=float(np.mean([x["finished"] for x in o])), eval_time=time.time() - t)


@torch.no_grad()
def kl_probe():
    """Sequence-level KL sample estimate for the final step, where no update follows: one batch of B x G samples from the current policy."""
    sync(); P = [tr[i] for i in random.sample(range(len(tr)), a.B)]; pr = [p for p in P for _ in range(a.G)]
    o = generate(gen, tok, pr, a.max_new, a.temp, a.gen_bs); seqs = [(x["prompt_ids"], x["comp_ids"]) for x in o]; kls = []
    for s in range(0, len(seqs), a.micro):
        sl = seqs[s:s + a.micro]; kls += (batch_logp(master, sl)[0] - batch_logp(ref, sl)[0]).tolist()
    torch.cuda.empty_cache(); return float(np.mean(kls))


order = list(range(len(tr))); random.shuffle(order); ptr = 0; t_start = time.time(); logf = open(a.out + "/log.jsonl", "a")
for step in range(0, a.steps + 1):
    rec = dict(step=step)
    if step % a.eval_every == 0: rec.update(do_eval(step))
    if step == a.steps or (time.time() - t_start) / 3600 > a.hours:
        rec.update(kl=kl_probe())
        if step > 0: torch.save({k: v.half() for k, v in master.state_dict().items()}, f"{a.out}/ckpt/step_{step}.pt")
        logf.write(json.dumps(rec) + "\n"); logf.flush(); break
    if step % a.ckpt_every == 0 and step > 0:
        torch.save({k: v.half() for k, v in master.state_dict().items()}, f"{a.out}/ckpt/step_{step}.pt")
    t0 = time.time()
    if ptr + a.B > len(order): random.shuffle(order); ptr = 0
    P = [tr[i] for i in order[ptr:ptr + a.B]]; ptr += a.B
    sync(); pr = [p for p in P for _ in range(a.G)]
    o = generate(gen, tok, pr, a.max_new, a.temp, a.gen_bs); torch.cuda.empty_cache(); t1 = time.time()
    texts = [x["text"] for x in o]; nt0 = prox.n_trunc; r = np.array(pscore(pr, texts), dtype=np.float64); ptrunc = (prox.n_trunc - nt0) / len(pr); torch.cuda.empty_cache(); t2 = time.time()
    r = np.nan_to_num(r, nan=0.0)
    R = r.reshape(a.B, a.G); adv = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-3); adv = adv.reshape(-1)
    adv[np.repeat(R.std(1) < 1e-6, a.G)] = 0.0
    seqs = [(x["prompt_ids"], x["comp_ids"]) for x in o]; kls = []; ktok = 0.0; opt.zero_grad(); nseq = len(seqs)
    for s in range(0, nseq, a.micro):
        sl = seqs[s:s + a.micro]; lp, n, lpt = batch_logp(master, sl)
        with torch.no_grad(): lr_, _, lrt = batch_logp(ref, sl)
        kls += (lp.detach() - lr_).tolist()
        A = torch.tensor(adv[s:s + a.micro], dtype=torch.float32).cuda()
        loss = -(A * lp).sum() / (nseq * a.max_new)
        if a.beta > 0:   # per-token k3 estimator of KL(pi||ref), summed over completion tokens, same normalisation as the PG term
            d = lrt - lpt; k3 = torch.exp(d) - d - 1; loss = loss + a.beta * k3.sum() / (nseq * a.max_new); ktok += k3.sum().item()
        scaler.scale(loss).backward()
    scaler.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(master.parameters(), 1.0); scaler.step(opt); scaler.update()
    fn, fs, fr = cross_prompt_rep(pr, texts)
    rec.update(proxy=float(r.mean()), proxy_sd_within=float(R.std(1).mean()), kl=float(np.mean(kls)), len=float(np.mean([len(c) for _, c in seqs])),
               fin=float(np.mean([x["finished"] for x in o])), rep3=fn, repsent=fs, repfrac=fr, gn=float(gn), ktok=ktok / nseq,
               prompt_trunc=float(np.mean([len(x["prompt_ids"]) >= 512 for x in o])), proxy_trunc=ptrunc,
               **({"gj_mass": prox.last_mass, "gj_mass_lt05": prox.last_mass_lo} if a.proxy == "gj" else {}), t_gen=t1 - t0, t_prox=t2 - t1, t_step=time.time() - t0)
    logf.write(json.dumps(rec) + "\n"); logf.flush()
    if step % 50 == 0:
        with open(f"{a.out}/samples/train_{step:05d}.jsonl", "w") as f:
            for p, x, s in list(zip(pr, o, r))[:40:4]: f.write(json.dumps(dict(prompt=p, text=x["text"], proxy=float(s))) + "\n")
    if step % 10 == 0: print(rec, flush=True)
print("done", flush=True)
