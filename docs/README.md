# GPT-OSS-20B RTX 5090 Inference Metrics

**Status: Phase 1 — setup + metrics working; write-up and benchmark sweep in progress**

This project runs [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) locally on a single
NVIDIA RTX 5090 using TensorRT-LLM, and serves it through a lightweight gateway proxy behind a
browser-based console. The console streams responses and displays live inference metrics: TTFT,
throughput, total latency, output tokens, and a full VRAM breakdown including KV-cache utilization.

It also includes two standalone analysis tools:
- `metrics_probe.py` — independently measures TTFT/TPOT/latency/VRAM against the raw server (Phase 1).
- `moe_probe.py` — profiles which experts fire, per layer, for a prompt (mixture-of-experts analysis).

---

## What it measures

- **TTFT** — time to first token
- **Throughput / TPOT** — decode speed (tokens/sec, and per-token latency)
- **Total latency** — end-to-end request time
- **Output tokens** — server-reported completion token count
- **VRAM breakdown** — weights, KV-cache pool, KV in use, activation/other, free
- **KV-cache utilization** — used vs. reserved blocks, live during generation
- **Expert activation** — per-expert, per-layer routing counts (via `moe_probe.py`)

---

## Hardware / software

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 5090 (Blackwell), ~31.3–31.8 GiB VRAM |
| Driver | 580.126.09 |
| CUDA | 13.0 |
| Runtime | TensorRT-LLM container `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc0` |
| Backend | PyTorch backend (`--backend pytorch`) |
| Model | `openai/gpt-oss-20b`, MXFP4, 3 safetensors shards (~12.8 GiB weights) |
| Serving | `trtllm-serve` (OpenAI-compatible) on :8000 |
| Gateway | `proxy.py` on :9000 (serves UI, streams API, computes `/vram`) |
| UI | `gpt-oss-console.html` (single-file browser console) |

Model config (from `config.json`): 24 layers, 8 KV heads, head-dim 64, 32 tokens/block,
KV pool ~8,819 blocks (~282K token capacity). MoE: see `config.json` for expert count and
experts-per-token (top-k).

---

## Architecture

```
browser (console UI)
      │  http://localhost:9000
      ▼
proxy.py  (gateway, :9000)
      ├─ serves gpt-oss-console.html
      ├─ /v1/*  /health         → forwarded to trtllm-serve (streamed, token-by-token)
      ├─ /metrics /perf_metrics  → forwarded to trtllm-serve
      └─ /vram                   → computed locally:
                                     weights = sum of *.safetensors
                                     KV pool = config.json dims × block count
                                     totals  = nvidia-smi
                                     KV live = /metrics (background-polled, cached)
      │  http://127.0.0.1:8000
      ▼
trtllm-serve  (:8000, GPU)
```

Single origin: the browser only talks to the proxy, so chat, metrics, and VRAM all flow through
one server with no CORS complications.

---

## Quick start

Inside the container (started with `--gpus all -p 8000:8000 -p 9000:9000` and the model mounted):

```bash
# 1. enable engine metrics
cat > /workspace/gateway/extra_llm_options.yaml << 'EOF'
enable_iter_perf_stats: true
return_perf_metrics: true
perf_metrics_max_requests: 256
EOF

# 2. launch server + proxy together (ABSOLUTE paths matter — see "Gotchas")
MODEL_DIR=/workspace/gpt-oss-20b GATEWAY_DIR=/workspace/gateway bash launch.sh
```

Open **http://localhost:9000/gpt-oss-console.html** and set the console endpoint (Settings)
to `http://localhost:9000`.

Manual launch:

```bash
# server
trtllm-serve /workspace/gpt-oss-20b --host 0.0.0.0 --port 8000 \
  --backend pytorch --tp_size 1 --ep_size 1 --trust_remote_code \
  --extra_llm_api_options /workspace/gateway/extra_llm_options.yaml

# proxy
cd /workspace/gateway
MODEL_DIR=/workspace/gpt-oss-20b TRTLLM=http://127.0.0.1:8000 PORT=9000 python3 proxy.py
```

---

## Metric definitions

- `request_start` — time before the HTTP request is sent
- `first_token_time` — time the first non-empty generated content token arrives
- `end_time` — time the streaming response completes
- **TTFT** = `first_token_time − request_start`
- `decode_time` = `end_time − first_token_time`
- **TPOT** = `decode_time / max(output_tokens − 1, 1)`
- **total_latency** = `end_time − request_start`
- **peak_vram** = maximum GPU memory observed during the request

Note: gpt-oss is a reasoning model. A large apparent TTFT is usually the hidden reasoning phase
(analysis channel) generated before the visible answer, not slow prefill — account for this when
reading first-token latency. True engine-side TTFT is available from `/perf_metrics`
(`first_token_time − arrival_time`), which excludes network/proxy overhead.

---

## How the KV cache works (and how it's measured)

The KV cache is GPU memory that stores the key/value attention vectors for tokens already
processed, so the model doesn't recompute attention over the whole sequence at every decode step.
It is **separate** from the browser's conversation history:

- **Conversation history** (browser `messages` array) is application-level text, resent with each
  request so the model has context. Reloading the page clears it. It does not touch VRAM.
- **KV cache** (server VRAM) is engine-level attention state. It survives page reloads and is only
  wiped when `trtllm-serve` restarts.

### It's a rolling pool, not per-conversation

The engine has no concept of "conversations." The KV cache is a fixed pool of blocks
(~8,819 blocks × 32 tokens/block here). Requests claim free blocks; when the pool fills, the engine
evicts old blocks (LRU) to make room. Nothing is explicitly "reset" between chats — blocks are
continuously reused and overwritten based purely on incoming token content.

### Prefix reuse is token-hash matching

When a request arrives, the engine hashes each block of input tokens and looks it up: if a block
with the same fingerprint (same tokens, same position) is already cached, it's a **hit** and the KV
is reused; the moment tokens differ, reuse stops and the rest is computed fresh. This is why:

- Resending a growing conversation verbatim reuses the shared prefix each turn (prefix caching).
- Unrelated one-off prompts reuse almost nothing (tokens diverge immediately).
- It works with no session tracking — reuse is purely a function of matching token sequences.

The `/metrics` fields reflect this: `reusedBlocks`, `cacheHitRate`, `missedBlocks`,
`usedNumBlocks`/`maxNumBlocks` (utilization), `tokensPerBlock`.

### Measurement path

`/metrics` (iteration stats) is **drain-on-read** (each read empties the queue) and only populated
**during active generation**. To avoid losing the race, `proxy.py` background-polls `/metrics`,
caches the latest non-empty reading, and serves `/vram` from that cache. KV pool bytes are computed
from `config.json` dims (`layers × 2 × kv_heads × head_dim × tokens_per_block × dtype_bytes`) times
the block count; utilization comes straight from `usedNumBlocks / maxNumBlocks`.

---

## Expert profiling — `moe_probe.py`

GPT-OSS-20B is a mixture-of-experts (MoE) model: each MoE layer has a router that scores all experts
per token and routes each token to its top-k. `moe_probe.py` records how many times each expert is
selected at each layer for a prompt.

**Run it separately from trtllm-serve** — it loads the model in HuggingFace Transformers, which
exposes routing that TensorRT-LLM fuses away for speed. Stop the server first so VRAM is free.

```bash
# inspect the model's MoE structure (expert count, top-k, router module names)
python3 moe_probe.py --model /workspace/gpt-oss-20b --inspect

# profile a prompt → per-(layer,expert) counts
python3 moe_probe.py --model /workspace/gpt-oss-20b \
  --prompt "Explain how photosynthesis works." \
  --max-new-tokens 64 --out expert_counts.csv
```

Output: a per-layer summary (busiest experts) plus `expert_counts.csv` with `layer,expert,count`.

### How it works

1. The router outputs logits of shape `[num_tokens, num_experts]` — a score per expert per token.
2. `topk` reduces this to `[num_tokens, k]` — the expert indices each token was routed to.
3. Flattening (`reshape(-1)`) drops the token dimension; `bincount` tallies how often each expert
   index appears → per-expert counts for that layer.
4. This runs for every layer, accumulating into a `[num_layers × num_experts]` matrix.

Two capture strategies, tried in order: `output_router_logits=True` (clean, profiles the prompt in
one forward pass), or **forward hooks** on the router modules (fallback, and captures generation too
via `model.generate`). Routing is per-token; the counts are the sum of all per-token decisions,
kept separate per layer.

### Why HuggingFace, not TensorRT-LLM

TRT-LLM fuses MoE routing and expert dispatch into compiled CUDA kernels for speed, so there's no
intermediate Python tensor to hook and no `output_router_logits` flag. HuggingFace runs eager
PyTorch with the router as a distinct, inspectable module — the right tool for studying internals.
Speed is irrelevant for an occasional profiling run.

### Memory note

Transformers may dequantize the MXFP4 weights toward bf16, pushing a 20B model past 32 GB and
causing CUDA OOM. Mitigations:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation (and make sure
  trtllm-serve is stopped so the card is free).
- `--device cpu` with a short prompt / small `--max-new-tokens` — slow but always fits; fine for
  a one-off routing profile.
- Load quantized (keep MXFP4) if your Transformers version supports it — stays ~13 GiB.

---

## Repo layout

```
gpt-oss-20b-rtx5090/
  README.md
  plan.md
  context.md
  scripts/
    launch.sh              # starts trtllm-serve + proxy, clean shutdown
    proxy.py               # gateway: UI + API streaming + /vram
    metrics_probe.py       # standalone metrics probe (Phase 1 coding task)
    moe_probe.py           # per-layer expert-activation profiler
  gateway/
    gpt-oss-console.html   # browser console UI
    extra_llm_options.yaml # engine metrics config
  docs/
    phase1_writeup.md
    debugging_notes.md
  results/
    phase1_metrics.csv
    expert_counts.csv
    screenshots/
```

Do **not** commit model weights (`*.safetensors`) — they're ~13 GB and not yours to redistribute.
See `.gitignore`.

---

## Gotchas hit (and fixed)

- **TTFT looked like 6–10s** — the proxy buffered the whole response (`r.read()`) before forwarding.
  Fixed by streaming: read in a loop, flush each chunk. (First token now reflects reality; large
  values are the reasoning phase, confirmed via `/perf_metrics`.)
- **VRAM weights read 0.00** — relative `MODEL_DIR` (`./gpt-oss-20b` or `../gpt-oss-20b`) resolved
  wrong after the launch script's `cd` into the gateway dir, so the safetensors glob matched nothing.
  Fixed with absolute paths.
- **`/metrics` empty** — three layers: the proxy wasn't forwarding `/metrics` (404 from the file
  server), `/metrics` is drain-on-read, and it's only populated during active generation. Fixed by
  forwarding the path and background-polling + caching in the proxy. (`enable_iter_perf_stats` is a
  valid option in 1.3.0rc0 — confirmed via `TorchLlmArgs.model_fields`.)
- **Port 9000 "refused"** — the container wasn't started with `-p 9000:9000`; the proxy was fine
  inside. Republish the port at `docker run` time.
- **`NVML: Unknown Error` / server won't start** — container lost GPU access (often a host
  `daemon-reload` mid-session). Fixed by restarting the container with `--gpus all`.
- **Empty `proxy.log` but full `trtllm.log`** — Python block-buffers stdout under redirection; use
  `python3 -u` or run in the foreground to see prints.

---

## Limitations

Single RTX 5090, single gateway, single session, non-batched, preliminary measurements.
Not a concurrency or batching benchmark.