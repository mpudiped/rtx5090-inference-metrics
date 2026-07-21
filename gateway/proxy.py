#!/usr/bin/env python3
"""
All-in-one proxy for the GPT-OSS console. One origin, everything through it:

  GET  /                     -> serves files from this directory (the console HTML)
  GET  /gpt-oss-console.html -> the console
  GET  /vram                 -> computed here: weights + KV + nvidia-smi breakdown
  *    /v1/... /health        -> forwarded to trtllm-serve (streamed, token-by-token)
       /metrics /perf_metrics -> forwarded to trtllm-serve

Run it in the same directory as gpt-oss-console.html, ideally INSIDE the container
(so it can read the model dir, run nvidia-smi, and reach localhost:8000):

    python3 proxy.py

Then open  http://localhost:9000/gpt-oss-console.html  and set the console's
endpoint (Settings) to  http://localhost:9000  — chat, metrics, and VRAM all
flow through this one server.

Env vars:
  TRTLLM          target server (default http://127.0.0.1:8000)
  PORT            port to serve on (default 9000)
  MODEL_DIR       model folder for weights/config (default ./gpt-oss-20b)
  GPU_INDEX       GPU to query (default 0)
  KV_DTYPE_BYTES  bytes per KV element (default 2; set 1 for fp8 KV cache)
"""
import os, glob, json, subprocess, urllib.request, urllib.error
from http.server import SimpleHTTPRequestHandler, HTTPServer

TRTLLM         = os.environ.get("TRTLLM", "http://127.0.0.1:8000").rstrip("/")
PORT           = int(os.environ.get("PORT", "9000"))
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(_HERE, "..", "gpt-oss-20b"))
MODEL_DIR = os.path.abspath(MODEL_DIR)
print(f"[vram] MODEL_DIR = {MODEL_DIR}")
print(f"[vram] safetensors found = {len(glob.glob(os.path.join(MODEL_DIR, '**', '*.safetensors'), recursive=True))}")
GPU_INDEX      = os.environ.get("GPU_INDEX", "0")
KV_DTYPE_BYTES = int(os.environ.get("KV_DTYPE_BYTES", "2"))

FORWARD_PREFIXES = ("/v1",)
FORWARD_EXACT    = ("/health", "/metrics", "/perf_metrics")


# ---------------- VRAM computation (was the separate helper) ----------------
def weights_bytes():
    total, files = 0, 0
    for f in glob.glob(os.path.join(MODEL_DIR, "**", "*.safetensors"), recursive=True):
        try:
            total += os.path.getsize(f); files += 1
        except OSError:
            pass
    return total, files

def read_config():
    try:
        with open(os.path.join(MODEL_DIR, "config.json")) as fh:
            c = json.load(fh)
    except Exception:
        return None
    n_layers = c.get("num_hidden_layers")
    n_heads  = c.get("num_attention_heads")
    n_kv     = c.get("num_key_value_heads", n_heads)
    hidden   = c.get("hidden_size")
    head_dim = c.get("head_dim") or (hidden // n_heads if hidden and n_heads else None)
    if not (n_layers and n_kv and head_dim):
        return None
    return {"num_layers": n_layers, "num_kv_heads": n_kv, "head_dim": head_dim}

def smi_card():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,nounits,noheader", "-i", str(GPU_INDEX)],
            text=True).strip().splitlines()[0]
        u, t = (int(x.strip()) for x in out.split(","))
        return u * 1024**2, t * 1024**2
    except Exception:
        return None, None

def smi_process():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,nounits,noheader", "-i", str(GPU_INDEX)],
            text=True).strip().splitlines()
        best = 0
        for line in out:
            if line.strip():
                _pid, mem = (x.strip() for x in line.split(","))
                best = max(best, int(mem) * 1024**2)
        return best or None
    except Exception:
        return None

def kv_from_metrics():
    try:
        with urllib.request.urlopen(TRTLLM + "/metrics", timeout=2) as r:
            data = json.loads(r.read())
        arr = data if isinstance(data, list) else data.get("metrics", [])
        if not arr:
            return None
        latest = max(arr, key=lambda e: e.get("iter", 0))
        kv = latest.get("kvCacheStats", {})
        return {
            "used_blocks": kv.get("usedNumBlocks"), "free_blocks": kv.get("freeNumBlocks"),
            "max_blocks": kv.get("maxNumBlocks"), "tokens_per_block": kv.get("tokensPerBlock"),
            "cache_hit_rate": kv.get("cacheHitRate"), "gpu_mem_usage": latest.get("gpuMemUsage"),
            "active_requests": latest.get("numActiveRequests"), "iter_latency_ms": latest.get("iterLatencyMS"),
        }
    except Exception:
        return None

def build_vram():
    w_bytes, w_files = weights_bytes()
    cfg = read_config()
    card_used, card_total = smi_card()
    proc = smi_process()
    kv = kv_from_metrics()
    engine_total = (kv or {}).get("gpu_mem_usage") or proc

    block_bytes = kv_pool = kv_used = None
    if cfg and kv and kv.get("tokens_per_block") and kv.get("max_blocks"):
        block_bytes = (kv["tokens_per_block"] * cfg["num_layers"] * 2
                       * cfg["num_kv_heads"] * cfg["head_dim"] * KV_DTYPE_BYTES)
        kv_pool = block_bytes * kv["max_blocks"]
        if kv.get("used_blocks") is not None:
            kv_used = block_bytes * kv["used_blocks"]

    other = None
    if engine_total and kv_pool is not None:
        other = max(0, engine_total - w_bytes - kv_pool)

    util = None
    if kv and kv.get("max_blocks") and kv.get("used_blocks") is not None:
        util = kv["used_blocks"] / kv["max_blocks"]

    return {
        "weights_bytes": w_bytes, "weights_files": w_files,
        "engine_total_bytes": engine_total, "card_used_bytes": card_used,
        "card_total_bytes": card_total, "process_bytes": proc,
        "kv": kv, "kv_config": cfg, "kv_dtype_bytes": KV_DTYPE_BYTES,
        "block_bytes": block_bytes, "kv_pool_bytes": kv_pool,
        "kv_used_bytes": kv_used, "other_bytes": other, "kv_utilization": util,
    }


# ---------------- request handling ----------------
class H(SimpleHTTPRequestHandler):
    def _should_forward(self):
        p = self.path.split("?")[0]
        return p.startswith(FORWARD_PREFIXES) or p in FORWARD_EXACT

    def _serve_vram(self):
        try:
            body = json.dumps(build_vram()).encode(); code = 200
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode(); code = 500
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def _forward(self, body=None):
        url = TRTLLM + self.path
        req = urllib.request.Request(url, data=body, method=self.command)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            r = urllib.request.urlopen(req)
            self.send_response(r.status)
            self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            while True:                       # stream: forward chunks as they arrive
                chunk = r.read(64)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except urllib.error.HTTPError as e:
            self.send_response(e.code); self.end_headers(); self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502); self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/vram":
            return self._serve_vram()
        if self._should_forward():
            return self._forward()
        return super().do_GET()               # serve the console HTML from disk

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self._forward(self.rfile.read(length))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"proxy on :{PORT}  ->  trtllm {TRTLLM}  |  model {MODEL_DIR}  |  gpu {GPU_INDEX}")
    print(f"open  http://localhost:{PORT}/gpt-oss-console.html  and set the console endpoint to  http://localhost:{PORT}")
    HTTPServer(("0.0.0.0", PORT), H).serve_forever()