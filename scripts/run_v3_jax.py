#!/usr/bin/env python3
"""V3 experiment runner — JAX backend on TPU.

Runs KV-coupled generation with LRU and Online Credit policies using the pure JAX
inference module. Compatible with the same benchmark data (.jsonl) used by
run_experiment.py but avoids PyTorch entirely.

Usage:
    TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0 \
    ~/cfkve-jax-env/bin/python scripts/run_v3_jax.py \
        --model-id Qwen/Qwen2.5-3B-Instruct \
        --policy lru \
        --capacity 256 \
        --data-path data/longctx-bench.jsonl \
        --output /tmp/v3-q25-c256-lru.jsonl
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import string
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ─── Scoring (from run_experiment.py, self-contained) ────────────────────────

def _normalized_token_list(text: Any) -> list[str]:
    lowered = str(text).lower().strip()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    return [tok for tok in no_punct.split() if tok]


def _normalized_exact_match_text(text: Any) -> str:
    lowered = str(text).lower().strip()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    return "".join(no_punct.split())


def _token_f1_score(prediction: Any, reference: Any) -> float:
    pred_tokens = _normalized_token_list(prediction)
    ref_tokens = _normalized_token_list(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = collections.Counter(pred_tokens)
    ref_counts = collections.Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(ref_tokens))
    denom = precision + recall
    return float(2.0 * precision * recall / denom) if denom > 0 else 0.0


def _extract_row(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Extract (question, context, references) from a benchmark row.

    Returns a list of reference strings to support max-over-answers scoring
    (LongBench standard).
    """
    prompt = (
        row.get("prompt") or row.get("question") or row.get("query")
        or row.get("input") or row.get("instruction") or ""
    )
    context = (
        row.get("context") or row.get("passage") or row.get("article")
        or row.get("document") or ""
    )
    ref = (
        row.get("answer") or row.get("answers") or row.get("reference")
        or row.get("reference_answer") or row.get("gold") or row.get("target")
        or row.get("output") or row.get("label") or ""
    )
    refs: list[str] = []
    if isinstance(ref, list):
        refs = [str(v).strip() for v in ref if str(v).strip()]
    elif isinstance(ref, dict):
        val = ref.get("text") or ref.get("answer") or ""
        if str(val).strip():
            refs = [str(val).strip()]
    else:
        if str(ref).strip():
            refs = [str(ref).strip()]
    return str(prompt).strip(), str(context).strip(), refs


# ─── Structural Protection Wrapper ───────────────────────────────────────────

class ProtectedPolicyWrapper:
    """Wraps a baseline policy to exclude prefix/suffix positions from eviction."""

    def __init__(self, inner_policy, prefix_frac=0.10, suffix_frac=0.10, capacity=0):
        self.inner = inner_policy
        self.name = f"protected-{inner_policy.name}"
        self.prefix_frac = prefix_frac
        self.suffix_frac = suffix_frac
        self.capacity = capacity
        self._n_prefill = 0  # set externally before first eviction

    def select_evictions(self, blocks, evict_count, step):
        if evict_count <= 0 or not blocks:
            return []
        # Compute protection bounds using capacity (not len(blocks)) to prevent
        # overflow on the first massive eviction (prefill → capacity).
        cap = self.capacity if self.capacity > 0 else len(blocks)
        prefix_protect = max(4, int(cap * self.prefix_frac)) if self.prefix_frac > 0 else 0
        suffix_protect = max(4, int(cap * self.suffix_frac)) if self.suffix_frac > 0 else 0

        # Identify positions to protect: lowest N (prefix) and highest N (suffix)
        all_positions = sorted(int(k) for k in blocks.keys())
        protected = set()
        for p in all_positions[:prefix_protect]:
            protected.add(str(p))
        for p in all_positions[-suffix_protect:]:
            protected.add(str(p))

        # Filter blocks to only include unprotected positions
        eligible = {k: v for k, v in blocks.items() if k not in protected}
        if not eligible:
            return []
        # Clamp evict count to eligible set
        actual_evict = min(evict_count, len(eligible))
        return self.inner.select_evictions(eligible, actual_evict, step)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V3 JAX experiment runner")
    parser.add_argument("--model-id", required=True, help="HuggingFace model ID")
    parser.add_argument("--policy", required=True,
                        choices=["lru", "online_credit", "snapkv", "streamingllm", "h2o",
                    "snapkv_faithful", "h2o_faithful",
                    "adakv", "adakv_faithful", "adakv_native",
                    "quest", "quest_faithful", "quest_native", "random"])
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--evict-every", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-cache-len", type=int, default=2048)
    parser.add_argument("--max-prompt-len", type=int, default=None,
                        help="Max prompt tokens (default: same as --max-cache-len). "
                             "Set higher than --max-cache-len for eviction experiments "
                             "where the full prompt should be fed and eviction handles the rest.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--max-examples", type=int, default=9999)
    parser.add_argument("--output", required=True, help="Per-item JSONL output path")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--eager", action="store_true",
                        help="Use eager (non-JIT) decode for models that OOM during JIT compilation")
    parser.add_argument("--prefill-chunk-size", type=int, default=None,
                        help="Tokens per prefill chunk (default 256; use 64 for 7B models)")
    parser.add_argument("--demotion-ratio", type=float, default=0.0,
                        help="Fraction of KV budget reallocated from bf16 to int8 (0=pure eviction, 0.25=25%% budget as int8)")
    # Ablation flags (only apply to online_credit policy)
    parser.add_argument("--ablation-no-credit", action="store_true",
                        help="Disable credit learning (random credit scores)")
    parser.add_argument("--ablation-no-gating", action="store_true",
                        help="Disable uncertainty gating (always use credit policy)")
    parser.add_argument("--ablation-no-protection", action="store_true",
                        help="Disable prefix/suffix structural protection")
    parser.add_argument("--ablation-no-priors", action="store_true",
                        help="Disable prior initialization in estimator")
    parser.add_argument("--ablation-reset-per-item", action="store_true",
                        help="Reset estimator between items (no cross-item learning)")
    parser.add_argument("--protect-prefix-suffix", action="store_true",
                        help="Add structural protection (prefix/suffix guards) to baseline policies")
    parser.add_argument("--protect-frac", type=float, default=0.10,
                        help="Fraction of capacity to protect as prefix and suffix (default: 0.10 = 10%% each)")
    parser.add_argument("--protect-prefix-frac", type=float, default=None,
                        help="Override prefix protection fraction (default: use --protect-frac)")
    parser.add_argument("--protect-suffix-frac", type=float, default=None,
                        help="Override suffix protection fraction (default: use --protect-frac)")
    parser.add_argument("--static-suffix", action="store_true",
                        help="Use static suffix protection (pin to original prompt boundary) instead of dynamic")
    parser.add_argument("--system-message", type=str, default=None,
                        help="Override the default system message (for non-QA tasks like summarization)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file, skipping already-processed items")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0=greedy, >0=stochastic)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for stochastic decoding (used when temperature>0)")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Disable EOS early stopping and decode full max_new_tokens for latency isolation")
    parser.add_argument("--tensor-parallel", type=int, default=None,
                        help="Number of devices for tensor parallelism (e.g. 8 for v4-32 worker)")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking mode for models that support it (e.g. Qwen3.5)")
    parser.add_argument("--eviction-regime", default="decode", choices=["decode", "prefill"],
                        help="When to apply the eviction policy: 'decode' (default, post-prefill+per-step) "
                             "or 'prefill' (one-shot post-prefill only, matches PagedAttention semantics)")
    args = parser.parse_args()

    # ── RNG for stochastic decoding ───────────────────────────────────────
    _rng = np.random.default_rng(args.seed) if args.temperature > 0 else None
    if args.temperature > 0:
        print(f"Stochastic decoding: temperature={args.temperature}, seed={args.seed}")

    import jax

    # Optional multi-host TPU setup for pod slices.
    # When these env vars are set, each worker process joins the same JAX
    # distributed runtime so jax.devices() exposes global devices.
    dist_coord = os.environ.get("CFKVE_JAX_COORDINATOR_ADDRESS", "").strip()
    dist_count_raw = os.environ.get("CFKVE_JAX_NUM_PROCESSES", "1")
    dist_pid_raw = os.environ.get("CFKVE_JAX_PROCESS_ID", "0")
    if dist_coord:
        try:
            dist_count = int(dist_count_raw)
            dist_pid = int(dist_pid_raw)
        except ValueError as exc:
            raise ValueError(
                "Invalid distributed env: CFKVE_JAX_NUM_PROCESSES and "
                "CFKVE_JAX_PROCESS_ID must be integers"
            ) from exc
        if dist_count > 1:
            try:
                jax.distributed.initialize(
                    coordinator_address=dist_coord,
                    num_processes=dist_count,
                    process_id=dist_pid,
                )
                print(
                    "Distributed JAX initialized: "
                    f"coord={dist_coord} process={dist_pid}/{dist_count}"
                )
            except RuntimeError as exc:
                # Safe no-op for wrappers that may initialize before entering this script.
                if "already initialized" not in str(exc).lower():
                    raise
                print(f"Distributed JAX already initialized: {exc}")
    # Enable persistent compilation cache — eliminates ~400s JIT warmup on 2nd+ runs
    # of the same model/shape. The cache is keyed on the compiled function and hardware.
    _cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jax_compile_cache"))
    os.makedirs(_cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", _cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    print(f"JAX {jax.__version__} | Backend: {jax.default_backend()}")
    print(
        f"Process index/count: {jax.process_index()}/{jax.process_count()} | "
        f"Local devices: {jax.local_device_count()} | Global devices: {jax.device_count()}"
    )
    print(f"Devices: {jax.devices()}")
    print(f"Compilation cache: {_cache_dir}")

    # ── Load model ────────────────────────────────────────────────────────
    from counterfact_kv_eviction.jax_inference import load_jax_model_and_tokenizer
    t0 = time.perf_counter()
    model, tokenizer = load_jax_model_and_tokenizer(
        args.model_id, max_cache_len=args.max_cache_len, eager=args.eager,
        prefill_chunk_size=args.prefill_chunk_size,
        tp_size=args.tensor_parallel,
    )
    load_s = time.perf_counter() - t0
    print(f"Model loaded in {load_s:.1f}s")

    # ── JIT warmup ────────────────────────────────────────────────────────
    if args.eager:
        print("Eager mode: skipping JIT warmup")
    elif not args.skip_warmup:
        print("Warming up JIT compilation (prefill-chunk + decode) ...")
        warm_t0 = time.perf_counter()

        # Warm up with the EXACT shapes used during real inference:
        # 1) Prefill chunk of _PREFILL_CHUNK_SIZE tokens (the shape generators use)
        # 2) Decode with 1 token (+ attention scores, which generators always need)
        # Using shorter dummy lengths (e.g. 3 tokens) causes recompilation on the
        # first real sample, leading to multiple compilations in memory → host OOM.
        from counterfact_kv_eviction.jax_inference import _PREFILL_CHUNK_SIZE
        chunk_size = _PREFILL_CHUNK_SIZE
        dummy_ids = list(range(1, chunk_size + 1))  # exactly chunk_size tokens
        out = model(input_ids=dummy_ids, output_attentions=True, use_cache=True)
        pv = out.past_key_values
        del out  # free prefill compilation artifacts before decode compile
        gc.collect()

        # Decode warmup (seq_len=1)
        out = model(input_ids=[chunk_size + 1], past_key_values=pv,
                     output_attentions=True, use_cache=True)
        pv = out.past_key_values
        del out; gc.collect()

        total_warm = time.perf_counter() - warm_t0
        print(f"JIT warmup: {total_warm:.1f}s")
    else:
        print("Skipping JIT warmup (--skip-warmup)")

    # ── Create generator ──────────────────────────────────────────────────
    from counterfact_kv_eviction.kv_coupled_generator import KVCoupledQwen35Generator
    from counterfact_kv_eviction.policies import (
        LRUPolicy,
        SnapKVPolicy,
        SnapKVFaithfulPolicy,
        StreamingSinkLikePolicy,
        HeavyHitterH2OLikePolicy,
        H2OFaithfulPolicy,
        AdaKVPolicy,
        AdaKVFaithfulPolicy,
        AdaKVNativePolicy,
        QUESTPolicy,
        QUESTFaithfulPolicy,
        QUESTNativePolicy,
        RandomEvictionPolicy,
    )

    generator = None
    online_gen = None

    if args.policy == "online_credit":
        from counterfact_kv_eviction.online_credit_generator import OnlineCreditGenerator
        ablation_kwargs: dict = {}
        if args.ablation_no_gating:
            ablation_kwargs["force_credit_policy"] = True  # bypass blend/warmup/fallback
        if args.ablation_no_protection:
            ablation_kwargs["prefix_protect_frac"] = 0.0
            ablation_kwargs["suffix_protect_frac"] = 0.0
        if args.ablation_no_credit:
            ablation_kwargs["ablation_random_credit"] = True
        if args.ablation_no_priors:
            ablation_kwargs["ablation_no_priors"] = True
        online_gen = OnlineCreditGenerator(
            model=model,
            tokenizer=tokenizer,
            capacity=args.capacity,
            evict_every_n_steps=args.evict_every,
            demotion_ratio=args.demotion_ratio,
            **ablation_kwargs,
        )
    # For LRU, we create a fresh generator per item (matches run_experiment.py behavior)

    # ── Load data ─────────────────────────────────────────────────────────
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"ERROR: data path not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(rows)} benchmark rows from {data_path}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    scored = 0
    exact_matches = 0
    f1_sum = 0.0
    total_evictions = 0
    per_item: list[dict[str, Any]] = []
    total_tokens_generated = 0
    total_decode_time_s = 0.0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: read already-completed item indices
    completed_indices: set[int] = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    completed_indices.add(rec["idx"])
        print(f"Resume: {len(completed_indices)} items already completed, skipping them")
        outf = out_path.open("a", encoding="utf-8")
        scored = len(completed_indices)
        f1_sum = 0.0  # Will only aggregate new items for progress display
    else:
        outf = out_path.open("w", encoding="utf-8")

    # Use the HF tokenizer directly for encoding
    hf_tok = tokenizer._tok

    # Collect stop token IDs for early stopping (instruct models end turns
    # with special tokens like <|im_end|> for Qwen or <|end|> for Phi).
    _eos_ids: set[int] = set()
    if hf_tok.eos_token_id is not None:
        _eos_ids.add(hf_tok.eos_token_id)
    for name in ["<|im_end|>", "<|end|>", "<|endoftext|>"]:
        tid = hf_tok.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid != hf_tok.unk_token_id:
            _eos_ids.add(tid)
    if args.ignore_eos:
        _eos_ids = set()
        print("EOS stop tokens: disabled (--ignore-eos)")
    else:
        print(f"EOS stop tokens: {_eos_ids}")

    # ── Compute chat-template overhead once ───────────────────────────────
    # Instruct-tuned models need their chat template for coherent generation.
    # We measure the fixed token overhead (system msg + template markers) so
    # we can allocate max budget to context while preserving the question.
    _SYS_MSG = args.system_message or "Answer the question based on the given passage. Only give a short factual answer."
    _overhead_msgs = [
        {"role": "system", "content": _SYS_MSG},
        {"role": "user", "content": ""},
    ]
    _chat_template_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if args.no_think:
        _chat_template_kwargs["enable_thinking"] = False
        print("Thinking mode DISABLED (--no-think)")
    _overhead_text = hf_tok.apply_chat_template(
        _overhead_msgs, **_chat_template_kwargs,
    )
    _template_overhead = len(hf_tok(_overhead_text, add_special_tokens=False)["input_ids"])
    gc_batch_size = 20
    items_processed = 0

    for idx, row in enumerate(rows):
        if scored >= args.max_examples:
            break
        if idx in completed_indices:
            continue

        question, context, references = _extract_row(row)
        items_processed += 1

        if (not question and not context) or not references:
            continue

        # Build token sequence using the model's chat template.
        # Truncate context from the LEFT to preserve the question, then wrap
        # in the instruct template so the model knows to answer concisely.
        _effective_prompt_limit = (args.max_prompt_len or args.max_cache_len) - args.max_new_tokens

        if context and question:
            q_part = f"\n\nQuestion: {question}"
            q_ids = hf_tok(q_part, add_special_tokens=False)["input_ids"]
            ctx_prefix = "Passage: "
            ctx_prefix_ids = hf_tok(ctx_prefix, add_special_tokens=False)["input_ids"]
            ctx_ids = hf_tok(context, add_special_tokens=False)["input_ids"]

            budget = _effective_prompt_limit - _template_overhead - len(q_ids) - len(ctx_prefix_ids)
            if budget <= 0:
                user_content = f"Question: {question}"
            else:
                if len(ctx_ids) > budget:
                    ctx_ids = ctx_ids[-budget:]
                ctx_text = hf_tok.decode(ctx_ids, skip_special_tokens=True)
                user_content = f"Passage: {ctx_text}{q_part}"
        else:
            user_content = question or context

        msgs = [
            {"role": "system", "content": _SYS_MSG},
            {"role": "user", "content": user_content},
        ]
        chat_text = hf_tok.apply_chat_template(
            msgs, **_chat_template_kwargs,
        )
        token_ids = hf_tok(
            chat_text, add_special_tokens=False,
            max_length=_effective_prompt_limit, truncation=True,
        )["input_ids"]

        if not token_ids:
            continue

        input_ids = [int(t) for t in token_ids]
        n_prompt = len(input_ids)

        print(f"  [{scored+1}] {n_prompt} prompt tokens, ", end="", flush=True)
        t_start = time.perf_counter()

        if args.policy == "online_credit":
            if args.ablation_reset_per_item:
                from counterfact_kv_eviction.estimator import OnlineCausalCreditEstimator
                online_gen.estimator = OnlineCausalCreditEstimator(
                    warmup_min_updates=online_gen.estimator.warmup_min_updates,
                )
            result = online_gen.generate_with_online_credit(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=_eos_ids,
                temperature=args.temperature,
                rng=_rng,
            )
        else:
            policy_map = {
                "lru": lambda: LRUPolicy(),
                "snapkv": lambda: SnapKVPolicy(),
                "snapkv_faithful": lambda: SnapKVFaithfulPolicy(),
                "streamingllm": lambda: StreamingSinkLikePolicy(
                    sink_size=4,
                    recent_window_size=max(0, args.capacity - 4),
                ),
                "h2o": lambda: HeavyHitterH2OLikePolicy(),
                "h2o_faithful": lambda: H2OFaithfulPolicy(),
                "adakv": lambda: AdaKVPolicy(),
                "adakv_faithful": lambda: AdaKVFaithfulPolicy(static_suffix_mode=args.static_suffix),
                "adakv_native": lambda: AdaKVNativePolicy(static_suffix_mode=args.static_suffix),
                "quest": lambda: QUESTPolicy(),
                "quest_faithful": lambda: QUESTFaithfulPolicy(static_suffix_mode=args.static_suffix),
                "quest_native": lambda: QUESTNativePolicy(static_suffix_mode=args.static_suffix),
                "random": lambda: RandomEvictionPolicy(),
            }
            policy_inst = policy_map[args.policy]()
            if args.protect_prefix_suffix:
                pfrac = args.protect_prefix_frac if args.protect_prefix_frac is not None else args.protect_frac
                sfrac = args.protect_suffix_frac if args.protect_suffix_frac is not None else args.protect_frac
                policy_inst = ProtectedPolicyWrapper(
                    policy_inst,
                    prefix_frac=pfrac,
                    suffix_frac=sfrac,
                    capacity=args.capacity,
                )
            gen = KVCoupledQwen35Generator(
                model=model,
                tokenizer=tokenizer,
                policy=policy_inst,
                capacity=args.capacity,
                evict_every_n_steps=args.evict_every,
                n_kv_heads=getattr(getattr(model, 'config', None),
                                   'num_key_value_heads', 0),
                eviction_regime=args.eviction_regime,
            )
            result = gen.generate_with_kv_control(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=_eos_ids,
                temperature=args.temperature,
                rng=_rng,
            )

        elapsed_s = time.perf_counter() - t_start
        n_generated = len(result.generated_token_ids) - n_prompt
        total_tokens_generated += n_generated
        total_decode_time_s += elapsed_s

        prediction_ids = result.generated_token_ids[n_prompt:]
        prediction = hf_tok.decode(prediction_ids, skip_special_tokens=True)

        # Max-over-answers scoring (LongBench standard)
        em = max(
            (1 if _normalized_exact_match_text(prediction) == _normalized_exact_match_text(r) else 0)
            for r in references
        )
        f1 = max(_token_f1_score(prediction, r) for r in references)

        if em:
            exact_matches += 1
        f1_sum += f1
        total_evictions += int(result.eviction_count)
        scored += 1

        ms_per_tok = (elapsed_s * 1000 / max(n_generated, 1))
        print(f"{n_generated} gen tokens, "
              f"F1={f1:.4f}, EM={int(em)}, "
              f"evictions={result.eviction_count}, "
              f"{ms_per_tok:.1f}ms/tok, "
              f"{elapsed_s:.1f}s total")

        record = {
            "idx": idx,
            "n_prompt_tokens": n_prompt,
            "n_generated_tokens": n_generated,
            "elapsed_s": round(elapsed_s, 3),
            "ms_per_token": round(ms_per_tok, 2),
            "exact_match": int(em),
            "token_f1": round(f1, 6),
            "eviction_count": int(result.eviction_count),
            "positions_retained": int(getattr(result, "positions_retained", 0)),
            "prefill_position_count": int(getattr(result, "prefill_position_count", 0)),
            "faithful_union_retained_max": int(getattr(result, "faithful_union_retained_max", 0)),
            "faithful_union_overflow_max": int(getattr(result, "faithful_union_overflow_max", 0)),
            "faithful_union_overflow_steps": int(getattr(result, "faithful_union_overflow_steps", 0)),
            "demotion_count": int(getattr(result, "demotion_count", 0)),
            "gate_open_count": len(getattr(result, "gate_opened_steps", [])),
            "gate_fallback_count": len(getattr(result, "gate_fallback_steps", [])),
            "policy_call_count": len(result.policy_call_steps),
            "policy_compute_s": round(float(getattr(result, "policy_compute_time_s", 0.0)), 6),
            "prefill_wall_s": round(float(getattr(result, "prefill_wall_s", 0.0)), 4),
            "decode_phase_wall_s": round(float(getattr(result, "decode_phase_wall_s", 0.0)), 4),
            "decode_step_n": int(getattr(result, "decode_step_n", 0)),
            "decode_step_mean_ms": round(float(getattr(result, "decode_step_mean_ms", 0.0)), 4),
            "decode_step_p50_ms": round(float(getattr(result, "decode_step_p50_ms", 0.0)), 4),
            "decode_step_p95_ms": round(float(getattr(result, "decode_step_p95_ms", 0.0)), 4),
            "decode_step_p99_ms": round(float(getattr(result, "decode_step_p99_ms", 0.0)), 4),
            "prediction": prediction[:500],
            "reference": references[0][:500] if references else "",
        }
        per_item.append(record)
        outf.write(json.dumps(record) + "\n")
        outf.flush()

        # Free TPU HBM between items to prevent accumulation
        result = None
        if args.policy != "online_credit":
            gen = None
            policy_inst = None
        # Batch garbage collection: collect every N items
        if items_processed % gc_batch_size == 0:
            gc.collect()

    outf.close()

    # ── Aggregate (re-read output file for correct stats after resume) ────
    all_records = []
    with out_path.open("r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))
    scored = len(all_records)
    if scored == 0:
        print("ERROR: no rows scored", file=sys.stderr)
        sys.exit(1)

    f1_sum = sum(r["token_f1"] for r in all_records)
    exact_matches = sum(r["exact_match"] for r in all_records)
    total_evictions = sum(r["eviction_count"] for r in all_records)
    total_tokens_generated = sum(r["n_generated_tokens"] for r in all_records)

    avg_f1 = f1_sum / scored
    avg_em = exact_matches / scored
    total_decode_time_s_all = sum(r["elapsed_s"] for r in all_records)
    tokens_per_sec = total_tokens_generated / max(total_decode_time_s_all, 0.001)

    print(f"\n{'='*60}")
    print(f"Model: {args.model_id}")
    print(f"Policy: {args.policy} | Capacity: {args.capacity}")
    print(f"Items scored: {scored}")
    print(f"Token F1: {avg_f1:.4f}")
    print(f"Exact Match: {avg_em:.4f}")
    print(f"Total evictions: {total_evictions}")
    print(f"Total tokens generated: {total_tokens_generated}")
    print(f"Throughput: {tokens_per_sec:.1f} tok/s")
    print(f"Run ID: {args.run_id or 'none'}")
    print(f"Per-item output: {args.output}")
    print(f"{'='*60}")

    # Write summary to a companion .summary.json
    summary_path = out_path.with_suffix(".summary.json")
    summary = {
        "model_id": args.model_id,
        "policy": args.policy,
        "capacity": args.capacity,
        "demotion_ratio": args.demotion_ratio,
        "evict_every": args.evict_every,
        "max_new_tokens": args.max_new_tokens,
        "max_cache_len": args.max_cache_len,
        "max_prompt_len": args.max_prompt_len or args.max_cache_len,
        "data_path": str(args.data_path),
        "scored": scored,
        "token_f1": round(avg_f1, 6),
        "exact_match": round(avg_em, 6),
        "total_evictions": total_evictions,
        "total_tokens_generated": total_tokens_generated,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "run_id": args.run_id,
        "temperature": args.temperature,
        "seed": args.seed if args.temperature > 0 else None,
        "jax_backend": "tpu",
    }
    with summary_path.open("w") as sf:
        json.dump(summary, sf, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
