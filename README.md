# smol-overopt

Inspired by Harry Liuson's [ultraviolet](https://github.com/hliuson/ultraviolet).

A small testbed for reward-model overoptimization in the style of Gao, Schulman and Hilton (2022), "Scaling Laws for Reward Model Overoptimization". A SmolLM2-135M-Instruct policy is trained with GRPO against a learned reward model (SmolLM2-360M with a scalar head) that was fitted to the scores of an LLM quality judge. A held-out set of 300 HelpSteer2 prompts is generated and judged every 25 steps. The judge plays the role of the gold reward, the learned model is the proxy.

## What it is for

A small, fast setting for prototyping experiments on reward hacking and reward-model overoptimization with a chat policy: the policy optimizes a learned reward model while an LLM judge tracks the quality of its answers on held-out prompts, so the two can be compared over training. A run of 500 steps takes a few hours on one 11 GB GPU. It is meant for trying ideas (KL penalties, early stopping, reward ensembles, detecting what the policy learns to exploit, and so on) with small compute.

## Install

```
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

The versions in `requirements.txt` are the ones the reference results were produced with. One CUDA GPU with about 8 GB free is needed for training (peak GPU use is around 6 to 7 GB). Models are downloaded from the Hugging Face Hub on first use. Two prerequisites are optional and not in `requirements.txt`: `vllm` (only if you serve the judge locally, and it needs its own GPU capacity) and the `rsync` executable (only for `eval_daemon.py` with a `host:dir` source).

The reference results used HelpSteer2 at revision `990b2711a36180dd19d9c94b8627844866f8982a`, SmolLM2-135M-Instruct at `12fd25f77366fa6b3b4b768ec3050bf629380bac` and SmolLM2-360M-Instruct at `a10cc1512eabd3dde888204e902eca88bddb4951` (pass `revision=` to the Hub calls, or check out those revisions, to reproduce exactly).

## Pipeline

1. Prompt split (HelpSteer2, 6489 train and 300 eval prompts, deterministic):
   `python prep_prompts.py`
2. Sample starting-policy answers on train prompts (3000 prompts, 2 samples each):
   `python gen_samples.py --out data/base.jsonl` (`--seed`, default 0)
3. Label them with the quality judge (set the judge variables first, see below):
   `python label.py data/base.jsonl data/base.lab.jsonl`
4. Train the reward model on the judge scores (held-out prompt Spearman is printed at the end):
   `python train_rm.py --data data/base.lab.jsonl --out data/rm360.pt`
5. Train the policy with GRPO against the reward model (standard config, 500 steps, writes `runs/rm/`):
   `python grpo.py --out runs/rm`
6. In a second shell, label the eval generations as they appear (writes `runs/rm/quality_curve.jsonl`):
   `python eval_daemon.py runs/rm runs/rm`
7. Plot judged quality and reward against step and against sqrt(KL):
   `python plots.py runs/rm --out curves.png`
8. Optional analysis of what the policy learned to exploit, and early stopping regret:
   `python tools/discover.py runs/rm` and `python tools/early_stop.py runs/rm`

Quick check that the install works (5 GRPO steps, dummy reward, no judge): `sh tests/smoke.sh`.

`grpo.py` defaults are the standard configuration: reward model proxy, 16 prompts x 8 samples per step, 256 new tokens, temperature 1, lr 5e-6, no KL penalty, 500 steps, eval on 300 prompts every 25 steps. Use `--beta 0.05` for a per-token KL penalty (k3 estimator against the starting policy), `--seed` for seeds, `--proxy gj` to use a small local generative judge (Qwen3-0.6B, expected score over digit tokens) as the reward instead of the reward model. `--temp` has to stay at 1 (the policy gradient and the KL estimate assume samples from the policy itself). `--out` must be a new or empty directory unless `--overwrite` is given, which clears it first. Prompts longer than 512 tokens after chat formatting are dropped at load time (the count is printed) so that the reward model and the judge see exactly what the policy saw. Each run directory holds `log.jsonl` (per-step reward, sequence KL, length, completion rate, repetition statistics), `eval/step_XXXXX.jsonl` (written atomically), `samples/` and `ckpt/` (policy weights every 100 steps and at the last step).

`eval_daemon.py` accepts either a local run directory or `host:dir` as its first argument, in which case the eval files are pulled with rsync, so training and judging can happen on different machines. Answers whose label is missing are retried, and the curve file records `n_labelled` next to `n`. Each run has an id in `args.json` that is copied into its curve rows, so a curve file from another run is never mixed in. The curve file reports `quality` (mean of helpfulness and correctness, 0 to 4), `quality_finished` (same, over answers that ended with an end-of-turn token), `proxy` (mean learned reward in judge units), length and completion rate.

## Judge configuration

`judge.py` talks to any OpenAI-compatible chat endpoint and is configured only by environment variables:

```
export JUDGE_BASE_URL=http://localhost:8000/v1 JUDGE_MODEL=Qwen/Qwen3-8B JUDGE_API_KEY=none
```

`JUDGE_KEY_FILE=/path/to/file` can be used instead of `JUDGE_API_KEY` to read the key from a file at call time. A local server can be started with, for example, `vllm serve Qwen/Qwen3-8B` (this needs `vllm` installed separately). The endpoint must accept the `chat_template_kwargs` field in the request (vLLM does); if it rejects it, set `JUDGE_NO_THINK_KWARGS=1` to omit the field, and use a judge that does not emit a thinking block. Run `python judge.py` to check the connection (it grades one wrong and one right answer). The judge is asked for `Helpfulness: <0-4> Correctness: <0-4>`; the parse is strict (last occurrence of each label, exactly one digit 0 to 4) and the raw judge text is saved next to every label in `*.lab.jsonl`. To use another judge, change the environment variables; to change the rubric, edit `RUBRIC` in `judge.py`. The reward model has to be retrained on labels from the judge that will also be used for evaluation, and absolute quality levels depend on the judge. A second model can label the first 150 eval answers with `eval_daemon.py --second MODEL` (the model has to be served by the same endpoint; steps that are already labelled are backfilled), which `tools/early_stop.py` then uses as a referee.

## Caveats

- If the same judge labels the reward model's training data and scores the eval answers, the setup measures overoptimization of a distilled proxy of that judge, and a hack that fools the judge itself is invisible; the optional second judge is a partial check.
- With a 256-token limit many starting-policy answers are cut off, so the completion rate can change along training. `quality_finished` is logged for comparison, but the finished subset itself changes along training.
- A reward model trained only on starting-policy answers has seen few good answers; how the reward model's training data is chosen matters.
- The policy is small and weak, so absolute quality is low, and results from one policy size and one proxy architecture may not transfer.

## Credits

The setup follows Gao, Schulman and Hilton, "Scaling Laws for Reward Model Overoptimization" (2022, https://arxiv.org/abs/2210.10760). Prompts are from HelpSteer2 (Wang et al., 2024; license CC-BY-4.0; https://huggingface.co/datasets/nvidia/HelpSteer2), models are SmolLM2 (Allal et al., 2025; license Apache-2.0; https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), and the policy update is GRPO (Shao et al., DeepSeekMath, 2024; https://arxiv.org/abs/2402.03300). Redistributing the data or weights keeps those licenses' attribution requirements.

## License

The code is released under the MIT license (see `LICENSE`).
