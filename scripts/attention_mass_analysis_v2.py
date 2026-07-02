#!/usr/bin/env python3
"""Attention mass analysis v2 — proper full-attention-matrix measurement.

Uses HuggingFace's standard forward pass with output_attentions=True to get
actual (n_heads, seq_q, seq_k) softmax attention matrices per layer.

Measures:
  1. Per-position attention mass (catches attention sinks)
  2. Boundary-region aggregate mass (prefix + suffix)
  3. Per-layer breakdown
  4. Cross-model comparison when run on multiple models

This replaces v1 which used per-key MAX scores from chunked prefill —
a lossy summary that naturally converges to uniform under max-then-normalize.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def analyze_attention_single_item(
    attentions: tuple,   # tuple of (batch, n_heads, seq_q, seq_k) per layer
    seq_len: int,
    n_prot: int,
) -> dict:
    """Analyze full attention matrices for boundary vs middle mass.

    Args:
        attentions: tuple of tensors, one per layer, each (1, n_heads, seq_q, seq_k)
        seq_len: actual sequence length (may be < seq_k if padded)
        n_prot: number of protected positions at each end
    """
    n_layers = len(attentions)
    prefix_end = n_prot
    suffix_start = max(0, seq_len - n_prot)

    # Per-layer results
    layer_results = []

    # Also track per-position attention mass (averaged across layers)
    position_mass_accum = np.zeros(seq_len, dtype=np.float64)

    for li in range(n_layers):
        # (1, n_heads, seq_q, seq_k) -> (n_heads, seq_len, seq_len)
        attn = attentions[li][0, :, :seq_len, :seq_len].float().cpu().numpy()
        n_heads = attn.shape[0]

        # Average across heads -> (seq_len, seq_len)
        # attn[h, q, k] = softmax attention weight from query q to key k in head h
        head_avg = attn.mean(axis=0)  # (seq_len, seq_len)

        # For each query position, how much attention goes to boundary vs middle keys?
        # Focus on middle-position QUERIES (these are the positions at risk of eviction)
        middle_q_start = prefix_end
        middle_q_end = suffix_start
        if middle_q_end <= middle_q_start:
            continue

        # Attention from middle queries to all keys
        middle_q_attn = head_avg[middle_q_start:middle_q_end, :]  # (n_mid_q, seq_len)

        # Mass on prefix keys, suffix keys, middle keys
        prefix_mass = float(middle_q_attn[:, :prefix_end].sum()) / middle_q_attn.shape[0]
        suffix_mass = float(middle_q_attn[:, suffix_start:seq_len].sum()) / middle_q_attn.shape[0]
        middle_mass = float(middle_q_attn[:, prefix_end:suffix_start].sum()) / middle_q_attn.shape[0]

        # Also: attention from ALL queries to each key position (for position-level analysis)
        per_key_mass = head_avg.mean(axis=0)  # (seq_len,) — mean attention each key gets
        position_mass_accum += per_key_mass

        # Also specifically check position 0 (attention sink)
        pos0_mass_from_middle = float(middle_q_attn[:, 0].mean())

        layer_results.append({
            "layer": li,
            "prefix_mass": prefix_mass,
            "suffix_mass": suffix_mass,
            "middle_mass": middle_mass,
            "boundary_mass": prefix_mass + suffix_mass,
            "pos0_mass_from_middle": pos0_mass_from_middle,
        })

        # Free memory
        del attn, head_avg, middle_q_attn

    # Aggregate across layers
    if not layer_results:
        return {}

    position_mass_avg = position_mass_accum / n_layers  # per-position mean attention

    boundary_masses = [lr["boundary_mass"] for lr in layer_results]
    pos0_masses = [lr["pos0_mass_from_middle"] for lr in layer_results]

    expected_uniform_boundary = 2 * n_prot / seq_len
    expected_uniform_per_pos = 1.0 / seq_len

    mean_boundary = float(np.mean(boundary_masses))
    enrichment = mean_boundary / (expected_uniform_boundary + 1e-12)

    # Position 0 enrichment
    mean_pos0 = float(np.mean(pos0_masses))
    pos0_enrichment = mean_pos0 / (expected_uniform_per_pos + 1e-12)

    # Top-5 highest attention positions
    top5_idx = np.argsort(position_mass_avg)[-5:][::-1]
    top5 = [(int(i), float(position_mass_avg[i])) for i in top5_idx]

    return {
        "n_layers": n_layers,
        "seq_len": seq_len,
        "n_prot": n_prot,
        "expected_uniform_boundary": expected_uniform_boundary,
        "mean_boundary_mass": mean_boundary,
        "boundary_enrichment": enrichment,
        "mean_pos0_mass": mean_pos0,
        "pos0_enrichment": pos0_enrichment,
        "boundary_mass_per_layer": boundary_masses,
        "pos0_mass_per_layer": pos0_masses,
        "top5_attention_positions": top5,
        "layer_details": layer_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Attention mass analysis v2 (proper full matrices)")
    parser.add_argument("--model-id", type=str, required=True,
                        help="HuggingFace model ID")
    parser.add_argument("--data-path", type=str, required=True,
                        help="JSONL benchmark file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL path")
    parser.add_argument("--max-items", type=int, default=30,
                        help="Max items to process")
    parser.add_argument("--start-item", type=int, default=0,
                        help="First item index to process (for extending existing runs)")
    parser.add_argument("--max-seq-len", type=int, default=1024,
                        help="Max sequence length (controls memory; full attention is O(n^2))")
    parser.add_argument("--protect-frac", type=float, default=0.10,
                        help="Protection fraction (default 10%%)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu, cuda, or auto")
    parser.add_argument("--dtype", type=str, default="auto",
                        help="Model dtype: auto (fp32 on CPU, bf16 on GPU), float16, bfloat16, float32")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    print(f"Loading model: {args.model_id} on {args.device}")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if args.dtype == "auto":
        dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    else:
        dtype = dtype_map.get(args.dtype, torch.float32)
    print(f"Using dtype: {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",  # Need eager attention to get attention weights
    )
    if args.device != "cpu":
        model = model.to(args.device)
    model.eval()
    print(f"Model loaded. Config: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_attention_heads} Q heads, "
          f"{getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)} KV heads")

    # Load data
    items = []
    with open(args.data_path) as f:
        for line in f:
            items.append(json.loads(line))
    items = items[args.start_item:args.start_item + args.max_items]
    print(f"Loaded {len(items)} items (start_item={args.start_item})")

    # Chat template
    _SYS_MSG = "Answer the question based on the given passage. Only give a short factual answer."
    _chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        _chat_kwargs["enable_thinking"] = False
    except Exception:
        pass

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, item in enumerate(items):
        question = item.get("question", item.get("input", ""))
        context_text = item.get("context", "")

        # Build prompt
        user_content = f"{context_text}\n\nQuestion: {question}\nAnswer:"
        try:
            msgs = [
                {"role": "system", "content": _SYS_MSG},
                {"role": "user", "content": user_content},
            ]
            chat_text = tokenizer.apply_chat_template(msgs, **_chat_kwargs)
        except Exception:
            msgs = [{"role": "user", "content": user_content}]
            chat_text = tokenizer.apply_chat_template(msgs, **_chat_kwargs)

        input_ids = tokenizer(chat_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        seq_len = input_ids.shape[1]

        # Truncate if needed (full attention matrices are O(n^2) memory)
        if seq_len > args.max_seq_len:
            input_ids = input_ids[:, :args.max_seq_len]
            seq_len = args.max_seq_len

        if seq_len < 100:
            print(f"  [{idx}] Skipping (too short: {seq_len})")
            continue

        n_prot = max(1, int(seq_len * args.protect_frac))
        print(f"  [{idx}/{len(items)}] seq_len={seq_len}, n_prot={n_prot}", end="", flush=True)

        t0 = time.perf_counter()
        device = next(model.parameters()).device
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(device),
                output_attentions=True,
                return_dict=True,
            )
        elapsed = time.perf_counter() - t0

        analysis = analyze_attention_single_item(
            outputs.attentions, seq_len, n_prot
        )

        if analysis:
            analysis["item_idx"] = idx
            analysis["prefill_time_s"] = elapsed
            results.append(analysis)
            print(f" | boundary={analysis['mean_boundary_mass']:.4f} "
                  f"(expect {analysis['expected_uniform_boundary']:.4f}) "
                  f"| enrich={analysis['boundary_enrichment']:.3f}x "
                  f"| pos0_enrich={analysis['pos0_enrichment']:.1f}x "
                  f"| {elapsed:.1f}s")
        else:
            print(f" | no analysis")

        # Free attention tensors
        del outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write results
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    if results:
        enrichments = [r["boundary_enrichment"] for r in results]
        pos0_enrichments = [r["pos0_enrichment"] for r in results]
        boundary_masses = [r["mean_boundary_mass"] for r in results]
        expected = results[0]["expected_uniform_boundary"]

        print(f"\n{'='*60}")
        print(f"SUMMARY ({len(results)} items)")
        print(f"{'='*60}")
        print(f"Mean boundary mass:   {np.mean(boundary_masses):.4f} (expect uniform: {expected:.4f})")
        print(f"Boundary enrichment:  {np.mean(enrichments):.3f}x ± {np.std(enrichments):.3f}")
        print(f"Position 0 enrichment: {np.mean(pos0_enrichments):.1f}x ± {np.std(pos0_enrichments):.1f}")
        print(f"  (>1x = attention sink; =1x = no sink)")

        # Per-layer summary
        n_layers = results[0]["n_layers"]
        layer_enrichments = []
        for li in range(n_layers):
            layer_bm = [r["boundary_mass_per_layer"][li] for r in results if li < len(r["boundary_mass_per_layer"])]
            layer_enrichments.append(np.mean(layer_bm) / (expected + 1e-12))
        print(f"\nPer-layer boundary enrichment range: [{min(layer_enrichments):.3f}x, {max(layer_enrichments):.3f}x]")
        print(f"Layer with max enrichment: {np.argmax(layer_enrichments)} ({max(layer_enrichments):.3f}x)")
        print(f"Layer with min enrichment: {np.argmin(layer_enrichments)} ({min(layer_enrichments):.3f}x)")

        # Position 0 per-layer
        layer_pos0 = []
        for li in range(n_layers):
            lp = [r["pos0_mass_per_layer"][li] for r in results if li < len(r["pos0_mass_per_layer"])]
            per_pos = 1.0 / results[0]["seq_len"]
            layer_pos0.append(np.mean(lp) / (per_pos + 1e-12))
        print(f"\nPosition 0 per-layer enrichment range: [{min(layer_pos0):.1f}x, {max(layer_pos0):.1f}x]")
        print(f"Layer with strongest pos0 sink: {np.argmax(layer_pos0)} ({max(layer_pos0):.1f}x)")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
