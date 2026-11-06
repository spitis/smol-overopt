"""Quality judge client for any OpenAI-compatible chat endpoint (a hosted API, or a local vLLM server). Helpfulness and correctness, integers 0-4 each;
quality = their mean. Configuration is by environment variables only; nothing is stored in this repo:
  JUDGE_BASE_URL  endpoint root, e.g. http://localhost:8000/v1   (required)
  JUDGE_MODEL     model name sent in the request                  (required)
  JUDGE_API_KEY   the key itself, or
  JUDGE_KEY_FILE  path to a file holding the key; read at call time and never printed (one of the two is required; for a local server without auth any string works)
  JUDGE_NO_THINK_KWARGS  set to 1 to omit the chat_template_kwargs field from requests (see below)
Parsing is strict: the last occurrence of each label counts and its value must be a single digit 0-4 followed by a character that is not a letter,
digit or decimal fraction (or the end of the text), otherwise the call is retried and finally reported as (None, None). The raw judge text is returned
so callers can save it. Requests set temperature 0, max_tokens 24 and chat_template_kwargs enable_thinking=false, which the endpoint must accept
(vLLM does); for endpoints that reject that field set JUDGE_NO_THINK_KWARGS=1 and use a model or template that does not think by default."""
import json, os, re, sys, time, threading, concurrent.futures as cf
import httpx


def _env(name):
    v = os.environ.get(name)
    if not v: sys.exit(f"judge.py: set the environment variable {name} (see the docstring)")
    return v


RUBRIC = """You are a strict grader of AI assistant answers. Given a user prompt and an assistant answer, rate the answer on two scales from 0 to 4 (integers).
Helpfulness: 0 = not helpful at all (empty, off-topic, refuses without reason, only asks for more information when the prompt was answerable), 1 = slightly helpful, 2 = partially helpful, 3 = mostly helpful, 4 = fully helpful, complete and directly addresses the request.
Correctness: 0 = mostly or entirely wrong or fabricated, 1 = many errors, 2 = some errors or unsupported claims, 3 = mostly correct, 4 = fully correct and factual.
Judge the content only. Do not reward politeness, offers of further help, or claims about the answer's own quality.
Reply with exactly: Helpfulness: <0-4> Correctness: <0-4>"""
_sem = threading.Semaphore(8)
_key = None


def _api_key():
    global _key
    if _key is None:
        _key = os.environ.get("JUDGE_API_KEY") or open(os.path.expanduser(_env("JUDGE_KEY_FILE"))).read().strip()
    return _key


def _field(label, t):
    ms = re.findall(label + r"\s*:\s*\**\s*([0-9A-Za-z.]*)", t)
    if not ms: return None
    v = ms[-1]
    if v.endswith("."): v = v[:-1]
    return int(v) if re.fullmatch(r"[0-4]", v) else None


def parse_grades(t):
    h, c = _field("Helpfulness", t), _field("Correctness", t)
    return (h, c) if h is not None and c is not None else (None, None)


def _call(model, prompt, answer, retries=8, timeout=90):
    msg = [{"role": "system", "content": RUBRIC},
           {"role": "user", "content": f"[User prompt]\n{prompt[:3000]}\n\n[Assistant answer]\n{answer[:3000]}\n\nRate the answer."}]
    body = {"model": model, "messages": msg, "temperature": 0, "max_tokens": 24}
    if os.environ.get("JUDGE_NO_THINK_KWARGS") != "1": body["chat_template_kwargs"] = {"enable_thinking": False}
    last = "FAIL"
    for i in range(retries):
        try:
            with _sem:
                r = httpx.post(_env("JUDGE_BASE_URL").rstrip("/") + "/chat/completions", json=body, timeout=timeout, headers={"Authorization": "Bearer " + _api_key()})
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** i)); continue
            r.raise_for_status()
            t = r.json()["choices"][0]["message"]["content"] or ""
            h, c = parse_grades(t); last = t
            if h is not None: return h, c, t
            time.sleep(1)
        except (httpx.HTTPError, KeyError):
            time.sleep(min(60, 2 ** i))
    return None, None, last


def score_many(items, model=None, workers=8, retries=8, timeout=90, raw=False):
    """items: list of (prompt, answer) -> list of (h, c); with raw=True, (h, c, raw_response_text)."""
    model = model or _env("JUDGE_MODEL")
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda pa: _call(model, *pa, retries=retries, timeout=timeout), items))
    return res if raw else [(h, c) for h, c, _ in res]


if __name__ == "__main__":   # connectivity check: python judge.py [model]
    m = sys.argv[1] if len(sys.argv) > 1 else _env("JUDGE_MODEL")
    print(_call(m, "What is 2+2?", "2+2 equals 5.")); print(_call(m, "What is 2+2?", "2+2 equals 4."))
