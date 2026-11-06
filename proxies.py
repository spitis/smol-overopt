import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


def chat_text(tok, prompt, answer):
    return tok.apply_chat_template([{"role": "user", "content": prompt[:2500]},
                                    {"role": "assistant", "content": answer if answer else "(empty)"}], tokenize=False)


class RM(torch.nn.Module):
    """SmolLM2-360M-Instruct trunk + linear head on the last token. Outputs are in judge units (mu + sd * head)."""
    def __init__(self, name="HuggingFaceTB/SmolLM2-360M-Instruct", max_len=640):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(name); self.tok.padding_side = "left"
        self.trunk = AutoModel.from_pretrained(name, dtype=torch.float32)
        self.head = nn.Linear(self.trunk.config.hidden_size, 1); self.max_len = max_len; self.name = name
        self.mu, self.sd = 0.0, 1.0; self.n_trunc = 0

    def forward(self, ids, mask):
        h = self.trunk(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -1]   # left padded -> last token
        return self.head(h.to(self.head.weight.dtype)).float().squeeze(-1)

    def encode(self, prompts, answers, dev="cuda"):
        """Long inputs keep the first max_len/4 and the last 3*max_len/4 tokens (middle deletion); n_trunc counts them."""
        ids = [self.tok(chat_text(self.tok, p, a), add_special_tokens=False).input_ids for p, a in zip(prompts, answers)]
        self.n_trunc += sum(len(x) > self.max_len for x in ids)
        ids = [x if len(x) <= self.max_len else x[:self.max_len // 4] + x[-(self.max_len - self.max_len // 4):] for x in ids]
        L = max(len(x) for x in ids); pad = self.tok.pad_token_id
        I = torch.tensor([[pad] * (L - len(x)) + x for x in ids]); M = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids])
        return I.to(dev), M.to(dev)

    @torch.no_grad()
    def score(self, prompts, answers, bs=32):
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]) + len(answers[i])); res = [0.0] * len(prompts)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]; I, M = self.encode([prompts[i] for i in idx], [answers[i] for i in idx])
            with torch.autocast("cuda", dtype=torch.float16): r = self(I, M)
            for j, i in enumerate(idx): res[i] = r[j].item() * self.sd + self.mu
        return res


class GenJudge:
    """Optional local generative-judge reward: zero-shot Qwen3 small model, expected value over the digit tokens 0-4."""
    SYS = ("You are a strict grader of AI assistant answers. Rate the assistant's answer to the user prompt on a scale from 0 to 4, "
           "where the score is the average of helpfulness and correctness. Helpfulness: 0 = not helpful, 4 = fully helpful, complete and directly addresses the request. "
           "Correctness: 0 = mostly wrong or fabricated, 4 = fully correct and factual. Judge content only; do not reward politeness or claims about the answer itself. "
           "Reply with 'Score: ' followed by one digit 0-4.")

    def __init__(self, name="Qwen/Qwen3-0.6B", dtype=torch.float16, max_len=900):
        self.tok = AutoTokenizer.from_pretrained(name); self.tok.padding_side = "left"
        self.m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).cuda().eval(); self.max_len = max_len; self.n_trunc = 0
        self.last_mass = float("nan"); self.last_mass_lo = float("nan")
        self.dig = [self.tok.convert_tokens_to_ids(str(d)) for d in range(5)]

    def _ids(self, p, a):
        msgs = [{"role": "system", "content": self.SYS},
                {"role": "user", "content": f"[User prompt]\n{p[:2000]}\n\n[Assistant answer]\n{(a or '(empty)')[:2500]}\n\nRate the answer."}]
        t = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + "Score: "
        x = self.tok(t, add_special_tokens=False).input_ids
        if len(x) <= self.max_len: return x
        self.n_trunc += 1; return x[:self.max_len // 3] + x[-(self.max_len - self.max_len // 3):]

    @torch.no_grad()
    def score(self, prompts, answers, bs=32):
        ids = [self._ids(p, a) for p, a in zip(prompts, answers)]; order = sorted(range(len(ids)), key=lambda i: len(ids[i])); res = [0.0] * len(ids)
        pad = self.tok.pad_token_id; masses = []
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]; L = max(len(ids[i]) for i in idx)
            I = torch.tensor([[pad] * (L - len(ids[i])) + ids[i] for i in idx]).cuda(); M = torch.tensor([[0] * (L - len(ids[i])) + [1] * len(ids[i]) for i in idx]).cuda()
            h = self.m.model(input_ids=I, attention_mask=M, use_cache=False).last_hidden_state[:, -1]
            full = self.m.lm_head(h).float(); masses.append(torch.softmax(full, -1)[:, self.dig].sum(-1))
            p = torch.softmax(full[:, self.dig], -1); ev = (p * torch.arange(5, device=p.device)).sum(-1)
            for j, i in enumerate(idx): res[i] = ev[j].item()
        m = torch.cat(masses); self.last_mass = m.mean().item(); self.last_mass_lo = (m < 0.5).float().mean().item()
        return res


class DummyReward:
    """Smoke-test reward: prefers answers of about 300 characters. Needs no model and no judge."""
    n_trunc = 0

    def score(self, prompts, answers, bs=32):
        return [-abs(len(a) - 300) / 300.0 for a in answers]
