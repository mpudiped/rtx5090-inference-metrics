#!/usr/bin/env python3
"""
moe_probe.py — profile which experts fire, per layer, for a given prompt.

Loads gpt-oss-20b in HuggingFace Transformers (NOT trtllm-serve — TRT-LLM fuses/hides
MoE routing for speed), runs a prompt, and records how many times each expert is
selected at each MoE layer. Outputs a CSV of [layer, expert, count] plus a text summary.

Two capture strategies, tried in order:
  1. output_router_logits=True  — clean, if the model supports it.
  2. forward hooks on router modules — fallback, auto-discovered by name.

USAGE
  # first, inspect the model's MoE structure so we know what we're working with:
  python moe_probe.py --model /workspace/gpt-oss-20b --inspect

  # then profile a prompt:
  python moe_probe.py --model /workspace/gpt-oss-20b \
      --prompt "Explain how photosynthesis works." \
      --out expert_counts.csv

NOTES
  - gpt-oss-20b is large; if it won't fit in VRAM, device_map="auto" offloads to CPU
    (slower, but fine for analysis). Use --device cpu to force CPU.
  - Routing is per-token: counts aggregate over all prompt+generated tokens.
  - top-k experts fire per token per layer, so per-layer totals ≈ k × num_tokens.
"""
import argparse, csv, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_cfg(cfg, *names, default=None):
    for n in names:
        v = getattr(cfg, n, None)
        if v is not None:
            return v
    return default


def load_model(model_dir, device):
    print(f"[moe] loading tokenizer + model from {model_dir} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    kwargs = dict(trust_remote_code=True, torch_dtype="auto")
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    if device not in ("auto",):
        model = model.to(device)
    model.eval()
    return tok, model


def inspect(model):
    cfg = model.config
    n_layers = get_cfg(cfg, "num_hidden_layers")
    n_experts = get_cfg(cfg, "num_local_experts", "num_experts", "n_routed_experts")
    top_k = get_cfg(cfg, "num_experts_per_tok", "num_experts_per_token", "moe_top_k")
    print("=== config (MoE-relevant) ===")
    print(f"  architectures       : {get_cfg(cfg, 'architectures')}")
    print(f"  num_hidden_layers   : {n_layers}")
    print(f"  num experts         : {n_experts}")
    print(f"  experts per token(k): {top_k}")
    print("\n=== modules that look MoE-related (name : type) ===")
    seen = 0
    for name, mod in model.named_modules():
        low = name.lower()
        if any(s in low for s in ("router", "gate", "expert", "moe")):
            print(f"  {name}  :  {type(mod).__name__}")
            seen += 1
            if seen > 60:
                print("  ... (truncated)")
                break
    if seen == 0:
        print("  (none matched — paste the full module list and we'll adjust)")
    print("\nRun again without --inspect (add --prompt) to profile expert usage.")


def layer_index_from_name(name):
    """Pull the integer layer index out of a module path like model.layers.12.mlp.router"""
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return -1


def profile(model, tok, prompt, max_new_tokens, device):
    cfg = model.config
    n_layers = get_cfg(cfg, "num_hidden_layers")
    n_experts = get_cfg(cfg, "num_local_experts", "num_experts", "n_routed_experts")
    top_k = get_cfg(cfg, "num_experts_per_tok", "num_experts_per_token", "moe_top_k", default=1)
    if n_experts is None:
        print("[moe] could not determine expert count from config — run --inspect.", file=sys.stderr)
        sys.exit(1)

    counts = torch.zeros((n_layers, n_experts), dtype=torch.long)

    # ---- strategy 1: output_router_logits ----
    inputs = tok(prompt, return_tensors="pt")
    first_param_device = next(model.parameters()).device
    inputs = {k: v.to(first_param_device) for k, v in inputs.items()}

    used_router_logits = False
    try:
        with torch.no_grad():
            out = model(**inputs, output_router_logits=True, use_cache=False)
        rl = getattr(out, "router_logits", None)
        if rl is not None and len(rl) > 0 and rl[0] is not None:
            used_router_logits = True
            for i, logits in enumerate(rl):
                if logits is None:
                    continue
                # logits: [num_tokens, num_experts]
                idx = logits.topk(top_k, dim=-1).indices.reshape(-1)
                binc = torch.bincount(idx.cpu(), minlength=n_experts)
                counts[i % n_layers] += binc[:n_experts]
            print(f"[moe] captured routing via output_router_logits ({len(rl)} layers).", file=sys.stderr)
    except TypeError:
        pass  # model.forward doesn't accept output_router_logits
    except Exception as e:
        print(f"[moe] router_logits path failed ({e}); falling back to hooks.", file=sys.stderr)

    # ---- strategy 2: forward hooks on router modules ----
    if not used_router_logits:
        print("[moe] using forward hooks on router modules.", file=sys.stderr)
        handles = []

        def make_hook(layer_idx):
            def hook(module, inp, output):
                # output is usually router logits [tokens, num_experts], or a tuple
                logits = output[0] if isinstance(output, (tuple, list)) else output
                if not torch.is_tensor(logits):
                    return
                if logits.dim() > 2:
                    logits = logits.reshape(-1, logits.shape[-1])
                if logits.shape[-1] != n_experts:
                    return
                idx = logits.topk(top_k, dim=-1).indices.reshape(-1)
                binc = torch.bincount(idx.cpu(), minlength=n_experts)
                if 0 <= layer_idx < n_layers:
                    counts[layer_idx] += binc[:n_experts]
            return hook

        matched = 0
        for name, mod in model.named_modules():
            low = name.lower()
            # target the router/gate submodule specifically, not the whole MoE block
            if low.endswith("router") or low.endswith("gate") or "router" in low.split(".")[-1]:
                li = layer_index_from_name(name)
                handles.append(mod.register_forward_hook(make_hook(li)))
                matched += 1
        if matched == 0:
            print("[moe] no router modules matched by name — run --inspect and paste the list.", file=sys.stderr)
            sys.exit(1)
        print(f"[moe] hooked {matched} router modules.", file=sys.stderr)

        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        for h in handles:
            h.remove()

    return counts, n_layers, n_experts, top_k


def report(counts, n_layers, n_experts, top_k, out_csv):
    total = int(counts.sum().item())
    print(f"\n=== expert activation summary ===")
    print(f"layers={n_layers}  experts={n_experts}  top_k={top_k}  total_triggers={total}\n")

    # per-layer: show the busiest experts
    for L in range(n_layers):
        row = counts[L]
        if row.sum() == 0:
            continue  # dense (non-MoE) layer or unused
        top = torch.topk(row, min(5, n_experts))
        pairs = ", ".join(f"e{int(i)}={int(c)}" for c, i in zip(top.values, top.indices))
        busiest = f"busiest: {pairs}"
        print(f"layer {L:>2}: {int(row.sum()):>6} triggers | {busiest}")

    # write full CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "expert", "count"])
        for L in range(n_layers):
            for E in range(n_experts):
                c = int(counts[L, E].item())
                if c:
                    w.writerow([L, E, c])
    print(f"\n[moe] wrote per-(layer,expert) counts to {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the gpt-oss-20b folder")
    ap.add_argument("--prompt", default="Explain how photosynthesis works in two paragraphs.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device", default="auto", help="'auto' (device_map), 'cuda', or 'cpu'")
    ap.add_argument("--out", default="expert_counts.csv")
    ap.add_argument("--inspect", action="store_true", help="print MoE structure and exit")
    args = ap.parse_args()

    tok, model = load_model(args.model, args.device)

    if args.inspect:
        inspect(model)
        return

    counts, n_layers, n_experts, top_k = profile(
        model, tok, args.prompt, args.max_new_tokens, args.device)
    report(counts, n_layers, n_experts, top_k, args.out)


if __name__ == "__main__":
    main()