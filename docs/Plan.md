# Project Plan

## Where things stand

The hard technical work of Phase 1 is done: GPT-OSS-20B runs on the RTX 5090, serves through the
gateway proxy and browser console end-to-end, streams token-by-token, and reports live metrics
(TTFT, throughput, VRAM breakdown, KV utilization). The remaining Phase 1 work is packaging —
writing it up, running a clean benchmark sweep, and building the standalone probe — plus one open
polish item on live KV caching.

Status snapshot:

- Done: model running, proxy + UI end-to-end, streaming, weights/VRAM totals, `/metrics` flowing,
  TTFT bug fixed, VRAM path bugs fixed, option names confirmed (`enable_iter_perf_stats` valid in
  1.3.0rc0).
- In progress: KV panel populating reliably (background poller), benchmark sweep, write-ups.
- Not started: `metrics_probe.py`, results table, screenshots, Phase 2 plan detail, commit.

---

## Phase 1 — remaining tasks

### 1. Finish the KV live-metrics polish
- Add the background `/metrics` poller to `proxy.py` so `/vram` serves cached KV stats and stops
  losing the drain-on-read race (currently `kv` can come back null even during generation).
- Verify the console VRAM panel shows `blocks used`, `kv in use`, and utilization ticking during a
  prompt.

### 2. Write `metrics_probe.py` (main coding task — write it myself)
Standalone script that hits the model server directly (no proxy/UI) and validates the dashboard.
- Call the OpenAI-compatible endpoint with `stream: true`.
- Measure TTFT (timer before request → first non-empty content token).
- Measure total latency (→ stream complete).
- Compute TPOT = decode_time / max(output_tokens − 1, 1).
- Sample VRAM during the request via `nvidia-smi` (peak).
- Run a set of fixed prompts: short chat, 1K context, 4K context, caption-refinement.
- Write one CSV row per run to `results/phase1_metrics.csv`.

CSV columns:
`timestamp,model,config,prompt_name,prompt_chars,max_tokens,output_tokens,ttft_ms,tpot_ms,total_latency_ms,peak_vram_mb`

Example:
```
python scripts/metrics_probe.py \
  --url http://localhost:8000/v1/chat/completions \
  --model gpt-oss-20b --runs 3 --max-tokens 256 \
  --out results/phase1_metrics.csv
```

### 3. Run the benchmark sweep
- Fixed prompts × configs, several runs each, discard the first (cold-start compile) run.
- Record TTFT, TPOT, total latency, output tokens, peak VRAM into the results table.

### 4. Compare probe vs UI
- Run `metrics_probe.py` and compare its numbers against the console's telemetry for the same
  prompts. Confirm they're in the same ballpark (independent validation).

### 5. Capture screenshot
- Console showing generated output plus TTFT / throughput / VRAM panel.

### 6. Write the docs
- `docs/phase1_writeup.md` — goal, setup, metrics, implementation, bugs, results, limitations,
  next phase.
- `docs/debugging_notes.md` — the TTFT streaming/buffering bug and the VRAM path/working-directory
  + drain-on-read bug, each with root cause and fix.

### 7. Commit everything
- Code, docs, screenshots, results checked into the repo. Phase 1 closed.

### Results table template

| Config / Precision | Prompt type | Prompt tokens | Output tokens | TTFT ms | TPOT ms/tok | Total latency s | Peak VRAM GB | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MXFP4 (default) | Short chat | | | | | | | |
| MXFP4 (default) | 1K prompt | | | | | | | |
| MXFP4 (default) | 4K prompt | | | | | | | |
| MXFP4 (default) | Caption refinement | | | | | | | |

Label clearly: *single RTX 5090, single gateway, single session, non-batched preliminary numbers.*

---

## Phase 2 — speculative decoding

Phase 1 gives the baseline. Phase 2 uses it to ask a deeper question:

> **Can a small draft model reduce latency for GPT-OSS-20B without hurting output quality?**

### Approach
- Add a smaller **draft model** that proposes several tokens ahead; the 20B **target** verifies them
  in one pass, accepting the run of tokens that match. Accepted tokens skip full target decode steps,
  cutting latency when acceptance is high.
- Baseline = target-only decode (Phase 1 numbers). Comparison = draft-assisted decode.

### New metrics
- **Acceptance rate** — fraction of drafted tokens the target accepts.
- **Target-only vs draft-assisted latency** — TTFT and TPOT under both.
- **Speedup** — end-to-end and per-token.
- **Quality preservation** — outputs should be equivalent to target-only:
  - caption-refinement quality (task-specific)
  - hallucination / factuality spot-checks
  - sanity: greedy draft-assisted output should match greedy target-only output token-for-token.

### Open questions to resolve in Phase 2
- Which draft model pairs well with gpt-oss-20b (architecture/tokenizer compatibility)?
- Speculation length vs acceptance-rate tradeoff (how many tokens to draft per step).
- Does TensorRT-LLM's speculative decoding path support this model on the PyTorch backend, and what
  VRAM cost does the draft model add on top of the 20B on a 32 GiB card?

### Definition of done (Phase 2)
- Draft-assisted decode running against the same harness.
- Acceptance rate, latency, speedup, and quality numbers recorded and compared to the Phase 1
  baseline.
- A short write-up answering the latency-vs-quality question with data.