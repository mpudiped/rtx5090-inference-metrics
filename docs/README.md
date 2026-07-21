# GPT-OSS-20B RTX 5090 Inference Metrics

**Status: Phase 1 — setup + metrics working; write-up and benchmark sweep in progress**

This project runs [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) locally on a single
NVIDIA RTX 5090 using TensorRT-LLM, and serves it through a lightweight gateway proxy behind a
browser-based console. The console streams responses and displays live inference metrics: TTFT,
throughput, total latency, output tokens, and a full VRAM breakdown including KV-cache utilization.

## What it measures

- **TTFT** — time to first token
- **Throughput / TPOT** — decode speed (tokens/sec, and per-token latency)
- **Total latency** — end-to-end request time
- **Output tokens** — server-reported completion token count
- **VRAM breakdown** — weights, KV-cache pool, KV in use, activation/other, free
- **KV-cache utilization** — used vs. reserved blocks, live during generation

Metrics are sourced from the engine where possible (`/metrics` iteration stats, `/perf_metrics`
per-request timing) and cross-checked client-side.

## Hardware / software

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 5090 (Blackwell), ~31.8 GiB VRAM |
| Driver | 580.126.09 |
| CUDA | 13.0 |
| Runtime | TensorRT-LLM container `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc0` |
| Backend | PyTorch backend (`--backend pytorch`) |
| Model | `openai/gpt-oss-20b`, MXFP4, 3 safetensors shards (~12.8 GiB weights) |
| Serving | `trtllm-serve` (OpenAI-compatible) on :8000 |
| Gateway | `proxy.py` on :9000 (serves UI, streams API, computes `/vram`) |
| UI | `gpt-oss-console.html` (single-file browser console) |

Model attention config (from `config.json`): 24 layers, 8 KV heads, head-dim 64, 32 tokens/block,
KV pool ~8,819 blocks (~282K token capacity).

## Architecture

```
browser (console UI)
      │  http://localhost:9000
      ▼
proxy.py  (gateway, :9000)
      ├─ serves gpt-oss-console.html
      ├─ /v1/*  /health        → forwarded to trtllm-serve (streamed, token-by-token)
      ├─ /metrics /perf_metrics → forwarded to trtllm-serve
      └─ /vram                  → computed locally:
                                    weights   = sum of *.safetensors
                                    KV pool   = config.json dims × block count
                                    totals    = nvidia-smi
                                    KV live   = /metrics (background-polled, cached)
      │  http://127.0.0.1:8000
      ▼
trtllm-serve  (:8000, GPU)
```

Single origin: the browser only talks to the proxy, so chat, metrics, and VRAM all flow through
one server with no CORS complications.

## Quick start

Inside the TensorRT-LLM container (started with `--gpus all -p 8000:8000 -p 9000:9000` and the
model mounted):

```bash
# 1. enable engine metrics (extra_llm_options.yaml)
cat > /workspace/gateway/extra_llm_options.yaml << 'EOF'
enable_iter_perf_stats: true
return_perf_metrics: true
perf_metrics_max_requests: 256
EOF

# 2. launch server + proxy together (absolute paths matter)
MODEL_DIR=/workspace/gpt-oss-20b GATEWAY_DIR=/workspace/gateway bash launch.sh
```

Then open **http://localhost:9000/gpt-oss-console.html** and set the console endpoint (Settings)
to `http://localhost:9000`.

Manual launch (server and proxy separately):

```bash
# server
trtllm-serve /workspace/gpt-oss-20b --host 0.0.0.0 --port 8000 \
  --backend pytorch --tp_size 1 --ep_size 1 --trust_remote_code \
  --extra_llm_api_options /workspace/gateway/extra_llm_options.yaml

# proxy
cd /workspace/gateway
MODEL_DIR=/workspace/gpt-oss-20b TRTLLM=http://127.0.0.1:8000 PORT=9000 python3 proxy.py
```

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
  gateway/
    gpt-oss-console.html   # browser console UI
    extra_llm_options.yaml # engine metrics config
  docs/
    phase1_writeup.md
    debugging_notes.md
  results/
    phase1_results.md
    phase1_metrics.csv
    screenshots/
```

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
(analysis channel) generated before the visible answer, not slow prefill — worth accounting for
when interpreting first-token latency.

## Limitations

Single RTX 5090, single gateway, single session, non-batched, preliminary measurements.
Not a concurrency or batching benchmark.

## Debugging highlights

See `docs/debugging_notes.md`. Two systems bugs of note were fixed in Phase 1:

- **TTFT was ~6–10s and wrong** because the proxy buffered the entire response (`r.read()`) before
  forwarding, so the browser received all tokens at once. Fixed by streaming: read in a loop and
  flush each chunk, so first-token timing reflects reality.
- **VRAM stats were empty** due to a chain of path/working-directory issues (relative `MODEL_DIR`
  resolving wrong across a `cd`, the proxy not forwarding `/metrics`, and `/metrics` being
  drain-on-read + only populated during active generation). Fixed with absolute paths, endpoint
  forwarding, and a background poller that caches the latest KV reading.