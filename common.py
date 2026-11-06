import json, os, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_prompts(path):
    d = json.load(open(path))
    return d["train"], d["eval"]


MAX_PROMPT_TOKENS = 512


def filter_prompts(tok, prompts, name="prompts", max_len=MAX_PROMPT_TOKENS):
    """Drop prompts whose chat-formatted token length exceeds max_len (generate() would otherwise cut off their end, including the
    assistant marker, while the reward model and judge would see the full text)."""
    keep = [p for p in prompts if len(tok(fmt_prompt(tok, p), add_special_tokens=False).input_ids) <= max_len]
    print(f"{name}: dropped {len(prompts) - len(keep)} of {len(prompts)} prompts longer than {max_len} tokens", flush=True)
    return keep


def fmt_prompt(tok, p):
    return tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def generate(model, tok, prompts, max_new=256, temp=1.0, bs=128, top_p=1.0, eos_id=None):
    """Returns a list of dict(prompt_ids, comp_ids, text, finished) in input order. model: fp16 on cuda, eval mode."""
    eos_id = eos_id if eos_id is not None else tok.convert_tokens_to_ids("<|im_end|>")
    tok.padding_side = "left"
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([fmt_prompt(tok, prompts[i]) for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False, truncation=True, max_length=MAX_PROMPT_TOKENS).to(model.device)
        kw = dict(do_sample=temp > 0, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, eos_token_id=eos_id)
        if temp > 0:
            kw.update(temperature=temp, top_p=top_p, top_k=0)
        g = model.generate(**enc, **kw)
        comp = g[:, enc.input_ids.shape[1]:]
        for j, i in enumerate(idx):
            c = comp[j].tolist()
            fin = eos_id in c
            if fin:
                c = c[:c.index(eos_id) + 1]
            pid = enc.input_ids[j][enc.attention_mask[j].bool()].tolist()
            txt = tok.decode([t for t in c if t != eos_id], skip_special_tokens=True).strip()
            out[i] = dict(prompt_ids=pid, comp_ids=c, text=txt, finished=fin)
    return out


def ngrams(text, n=3):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def sentences(text):
    return {s.strip().lower() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 15}


def cross_prompt_rep(prompts, texts, base_thresh=3):
    """Fraction of answers with >=1 3-gram (and one sentence) that also occurs in answers to >=base_thresh (resp. 1) other
    prompts, plus the mean fraction of an answer's 3-grams that are common in that sense."""
    from collections import defaultdict
    ng = defaultdict(set); st = defaultdict(set)
    G = [ngrams(t) for t in texts]; S = [sentences(t) for t in texts]
    for i, p in enumerate(prompts):
        for g in G[i]: ng[g].add(p)
        for s in S[i]: st[s].add(p)
    fn = sum(any(len(ng[g] - {p}) >= base_thresh for g in G[i]) for i, p in enumerate(prompts)) / len(texts)
    fs = sum(any(len(st[s] - {p}) >= 1 for s in S[i]) for i, p in enumerate(prompts)) / len(texts)
    fr = sum(sum(len(ng[g] - {p}) >= base_thresh for g in G[i]) / max(1, len(G[i])) for i, p in enumerate(prompts)) / len(texts)
    return fn, fs, fr
