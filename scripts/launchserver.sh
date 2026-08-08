#trtllm-serve serve --host 0.0.0.0 --port 8000 --backend pytorch --trust_remote_code --config extra_llm_options.yaml ./gpt-oss-20b/

#!/usr/bin/env bash
# launch.sh — start trtllm-serve and the gateway proxy together, then stop both on Ctrl+C.
# Run this INSIDE the container (needs the model dir, nvidia-smi, and localhost:SERVE_PORT).
#
# Layout assumed (override any with env vars):
#   model    -> /models/gpt-oss-20b
#   gateway  -> /models/gateway   (contains proxy.py and gpt-oss-console.html)

set -euo pipefail

# ---------------- config (override via env) ----------------
MODEL_DIR="${MODEL_DIR:-/workspace/gpt-oss-20b}"
GATEWAY_DIR="${GATEWAY_DIR:-/workspace/gateway}"
SERVE_PORT="${SERVE_PORT:-8000}"
PROXY_PORT="${PROXY_PORT:-9000}"
GPU_INDEX="${GPU_INDEX:-0}"
KV_DTYPE_BYTES="${KV_DTYPE_BYTES:-2}"
# Optional: path to an extra_llm_api_options YAML (e.g. to enable perf metrics).
# Leave empty to launch without it.
EXTRA_OPTS="${EXTRA_OPTS:-/workspace/extra_llm_options.yaml}"

LOG_DIR="${LOG_DIR:-$GATEWAY_DIR/logs}"
mkdir -p "$LOG_DIR"
TRTLLM_LOG="trtllm.log"
PROXY_LOG="proxy.log"

# ---------------- sanity checks ----------------
[ -d "$MODEL_DIR" ]   || { echo "[launch] MODEL_DIR not found: $MODEL_DIR"; exit 1; }
[ -f "$GATEWAY_DIR/proxy.py" ] || { echo "[launch] proxy.py not found in $GATEWAY_DIR"; exit 1; }
if ! ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
  echo "[launch] WARNING: no .safetensors in $MODEL_DIR — weights will read as 0."
fi

# ---------------- clean shutdown of both ----------------
PIDS=()
cleanup(){
  echo ""
  echo "[launch] shutting down..."
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo "[launch] stopped."
}
trap cleanup EXIT INT TERM

# ---------------- 1. trtllm-serve ----------------
echo "[launch] starting trtllm-serve on :$SERVE_PORT  (model: $MODEL_DIR)"
serve_args=(
  "$MODEL_DIR"
  --host 0.0.0.0 --port "$SERVE_PORT"
  --backend pytorch --tp_size 1 --ep_size 1
  --trust_remote_code
)
[ -n "$EXTRA_OPTS" ] && serve_args+=(--extra_llm_api_options "$EXTRA_OPTS")

trtllm-serve "${serve_args[@]}" > "$GATEWAY_DIR/logs/$TRTLLM_LOG" 2>&1 &
SERVE_PID=$!
PIDS+=("$SERVE_PID")
echo "[launch] trtllm-serve pid $SERVE_PID  (logs: $GATEWAY_DIR/logs/$TRTLLM_LOG)"

# stream the server log so you can watch it load
tail -n +1 -F "$GATEWAY_DIR/logs/$TRTLLM_LOG" &
TAIL_PID=$!
PIDS+=("$TAIL_PID")

# ---------------- 2. wait until the server is actually serving ----------------
echo "[launch] waiting for /health (first load can take a while)..."
for i in $(seq 1 900); do   # up to 15 min for cold-start compile
  if curl -sf "http://127.0.0.1:$SERVE_PORT/health" >/dev/null 2>&1; then
    echo "[launch] server ready after ${i}s"
    break
  fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "[launch] trtllm-serve exited during startup — see $GATEWAY_DIR/logs/$TRTLLM_LOG above."
    exit 1
  fi
  sleep 1
done

# ---------------- 3. gateway proxy ----------------
echo "[launch] starting proxy on :$PROXY_PORT"
cd "$GATEWAY_DIR"
TRTLLM="http://127.0.0.1:$SERVE_PORT" \
MODEL_DIR="$MODEL_DIR" \
PORT="$PROXY_PORT" \
GPU_INDEX="$GPU_INDEX" \
KV_DTYPE_BYTES="$KV_DTYPE_BYTES" \
python3 -u "proxy.py" > "./logs/$PROXY_LOG" 2>&1 &
PROXY_PID=$!
PIDS+=("$PROXY_PID")
echo "[launch] proxy pid $PROXY_PID  (logs: ./logs/$PROXY_LOG)"

tail -n +1 -F "./logs/$PROXY_LOG" &
PIDS+=("$!")

echo ""
echo "[launch] READY  ->  http://localhost:$PROXY_PORT/gpt-oss-console.html"
echo "[launch] set the console endpoint (Settings) to  http://localhost:$PROXY_PORT"
echo "[launch] Ctrl+C to stop both."
echo ""

# exit (and trigger cleanup) as soon as either the server or proxy dies
while kill -0 "$SERVE_PID" 2>/dev/null && kill -0 "$PROXY_PID" 2>/dev/null; do
  sleep 2
done
echo "[launch] a process exited — stopping the other."