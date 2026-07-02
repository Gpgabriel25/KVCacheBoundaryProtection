"""Pure-JAX transformer inference for Qwen2, Qwen3.5, Phi3, Gemma3, and Llama architectures.

Loads weights from safetensors via the existing jax_safetensors_loader module
and implements a forward pass that returns logits, attention scores, and
KV cache — matching the interface expected by KVCoupledQwen35Generator.

Supported architectures:
  - Qwen3.5ForCausalLM  (Hybrid: Gated DeltaNet linear + GQA with partial RoPE)
  - Qwen2ForCausalLM    (GQA, separate Q/K/V with bias, separate gate/up MLP)
  - Phi3ForCausalLM      (MHA, fused QKV, fused gate_up MLP, no QKV bias)
  - LlamaForCausalLM     (GQA, separate Q/K/V without bias, separate gate/up MLP)
  - Gemma3ForCausalLM    (GQA, sandwich norm, GELU-tanh, per-layer RoPE, embed scaling)

Design goals:
  - Multi-chip TPU inference with tensor parallelism
  - bf16 throughout for TPU efficiency
  - Static KV cache shapes for XLA compilation friendliness
  - Returns per-key attention scores for eviction policy decisions
"""

from __future__ import annotations

from functools import partial
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None  # type: ignore[assignment]


def _ensure_numpy_float8_compat() -> None:
    """Expose float8 dtypes on numpy when provided by ml_dtypes.

    Some safetensors builds resolve float8 symbols from numpy directly.
    Older numpy builds in TPU images may miss these attributes.
    """
    if ml_dtypes is None:
        return
    for name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    ):
        if not hasattr(np, name) and hasattr(ml_dtypes, name):
            setattr(np, name, getattr(ml_dtypes, name))


_ensure_numpy_float8_compat()

try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]


# ─── Model configuration ──────────────────────────────────────────────────────

@dataclass
class TransformerConfig:
    """Architecture-agnostic transformer config."""

    arch: str  # "qwen2", "qwen3", "phi3", "llama", "gemma3", or "qwen3_5_hybrid"
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool
    max_position_embeddings: int
    # Phi3-specific
    fused_qkv: bool = False
    fused_gate_up: bool = False
    has_qkv_bias: bool = False
    # Hybrid architecture (Qwen3.5): per-layer type ("linear" or "full")
    layer_types: tuple[str, ...] | None = None
    partial_rotary_factor: float = 1.0  # fraction of head_dim for RoPE
    attn_output_gate: bool = False  # GQA layers use g_proj output gate
    # Gated DeltaNet linear attention config
    linear_num_k_heads: int = 0
    linear_num_v_heads: int = 0
    linear_head_k_dim: int = 0
    linear_head_v_dim: int = 0
    linear_conv_kernel_dim: int = 0
    norm_residual_weight: bool = False  # Qwen3.5 uses (1+w) norm; others use w
    # RoPE scaling (Phi-3.5 LongRoPE or Llama-3.1 llama3)
    rope_scaling_type: str | None = None  # "longrope", "llama3", or None
    rope_long_factor: tuple[float, ...] | None = None
    rope_short_factor: tuple[float, ...] | None = None
    original_max_position_embeddings: int = 0  # base context length (switch threshold)
    # Llama-3.1 "llama3" RoPE scaling params
    rope_scaling_factor: float = 1.0  # context extension factor (e.g. 8.0)
    rope_scaling_low_freq_factor: float = 1.0
    rope_scaling_high_freq_factor: float = 1.0
    # Gemma3-specific
    sandwich_norm: bool = False  # pre+post norms around both attention and MLP
    hidden_activation: str = "silu"  # "silu" or "gelu_tanh"
    query_pre_attn_scalar: float = 0  # >0: use this instead of head_dim for attn scale
    gemma_embed_scale: bool = False  # multiply embeddings by sqrt(hidden_size)
    gemma3_layer_attn_types: tuple[str, ...] | None = None  # per-layer "sliding"/"global"
    gemma3_sliding_window: int | None = None
    gemma3_local_rope_theta: float = 10000.0
    gemma3_global_rope_theta: float = 1000000.0
    gemma3_global_rope_factor: float = 8.0  # linear position scaling for global layers

    @property
    def num_kv_layers(self) -> int:
        """Number of layers with traditional KV cache."""
        if self.layer_types is None:
            return self.num_hidden_layers
        return sum(1 for lt in self.layer_types if lt == "full")

    @property
    def num_linear_layers(self) -> int:
        """Number of Gated DeltaNet linear attention layers."""
        if self.layer_types is None:
            return 0
        return sum(1 for lt in self.layer_types if lt == "linear")

    @classmethod
    def from_hf_config(cls, config_path: str | Path) -> "TransformerConfig":
        """Load from a HuggingFace config.json file."""
        with open(config_path) as f:
            cfg = json.load(f)

        model_type = cfg.get("model_type", "")

        # Gemma3: multimodal wrapper — parse text_config and return early
        # The raw JSON text_config is sparse (only overrides); most fields use
        # Gemma3TextConfig class defaults which we hardcode here.
        if model_type == "gemma3" and "text_config" in cfg:
            tc = cfg["text_config"]
            n_layers = tc.get("num_hidden_layers", 34)
            n_heads = tc.get("num_attention_heads", 8)
            n_kv_heads = tc.get("num_key_value_heads", 4)
            h_size = tc.get("hidden_size", 2560)
            tc_head_dim = tc.get("head_dim", 256)  # Gemma3 default; NOT hidden_size//n_heads

            # Layer types: raw JSON may omit this; compute from pattern
            # (every sliding_window_pattern-th layer is full_attention)
            raw_lt = tc.get("layer_types", [])
            if raw_lt:
                g3_layer_types = tuple(
                    "global" if "full" in lt else "sliding"
                    for lt in raw_lt
                )
            else:
                sw_pattern = tc.get("sliding_window_pattern", 6)
                g3_layer_types = tuple(
                    "global" if (i + 1) % sw_pattern == 0 else "sliding"
                    for i in range(n_layers)
                )
            if len(g3_layer_types) != n_layers:
                raise ValueError(
                    f"layer_types length {len(g3_layer_types)} != "
                    f"num_hidden_layers {n_layers}"
                )

            # Per-layer RoPE: resolved rope_parameters may not be in raw JSON.
            # The raw JSON may have rope_scaling with just {factor, rope_type}.
            # Hardcode Gemma3 defaults for per-layer theta.
            rope_cfg = tc.get("rope_parameters", tc.get("rope_scaling", {}))
            local_theta = 10000.0
            global_theta = 1000000.0
            global_factor = 8.0
            if isinstance(rope_cfg, dict):
                if "sliding_attention" in rope_cfg:
                    local_theta = rope_cfg["sliding_attention"].get("rope_theta", 10000.0)
                    global_theta = rope_cfg.get("full_attention", {}).get("rope_theta", 1000000.0)
                    global_factor = rope_cfg.get("full_attention", {}).get("factor", 8.0)
                elif "factor" in rope_cfg:
                    # Simple {factor, rope_type} format — use Gemma3 defaults
                    global_factor = rope_cfg.get("factor", 8.0)

            return cls(
                arch="gemma3",
                vocab_size=tc.get("vocab_size", 262208),
                hidden_size=h_size,
                intermediate_size=tc.get("intermediate_size", 10240),
                num_attention_heads=n_heads,
                num_key_value_heads=n_kv_heads,
                num_hidden_layers=n_layers,
                rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
                rope_theta=global_theta,
                head_dim=tc_head_dim,
                tie_word_embeddings=tc.get("tie_word_embeddings", True),
                max_position_embeddings=tc.get("max_position_embeddings", 131072),
                fused_qkv=False,
                fused_gate_up=False,
                has_qkv_bias=False,
                norm_residual_weight=True,
                sandwich_norm=True,
                hidden_activation="gelu_tanh",
                query_pre_attn_scalar=float(tc.get("query_pre_attn_scalar", tc_head_dim)),
                gemma_embed_scale=True,
                gemma3_layer_attn_types=g3_layer_types if g3_layer_types else None,
                gemma3_sliding_window=int(tc.get("sliding_window", 1024)),
                gemma3_local_rope_theta=local_theta,
                gemma3_global_rope_theta=global_theta,
                gemma3_global_rope_factor=global_factor,
            )

        # Mistral3/Ministral3 wrap text config inside text_config.
        # Reuse the existing mistral/llama implementation for text-only decoding.
        if model_type in ("mistral3", "ministral3") and "text_config" in cfg:
            cfg = cfg["text_config"]
            model_type = cfg.get("model_type", "mistral")
            if model_type in ("mistral3", "ministral3"):
                model_type = "mistral"

        # Qwen3.5 wraps text config inside text_config
        if model_type == "qwen3_5" and "text_config" in cfg:
            cfg = cfg["text_config"]
            model_type = "qwen3_5_text"

        hidden = cfg["hidden_size"]
        n_heads = cfg["num_attention_heads"]
        head_dim = cfg.get("head_dim", hidden // n_heads)

        if model_type == "qwen3_5_text":
            # Hybrid architecture: DeltaNet linear + full GQA layers
            n_layers = cfg["num_hidden_layers"]
            # Parse layer_types from config, or infer standard [L,L,L,F]×N pattern
            raw_lt = cfg.get("layer_types", cfg.get("attention_types"))
            if raw_lt is not None:
                # Normalize: "linear_attention" → "linear", "full_attention" → "full"
                normalized = []
                for lt in raw_lt:
                    if "linear" in lt:
                        normalized.append("linear")
                    else:
                        normalized.append("full")
                layer_types = tuple(normalized)
            else:
                pattern = ("linear", "linear", "linear", "full")
                layer_types = pattern * (n_layers // 4)
                if n_layers % 4 != 0:
                    layer_types = layer_types + ("linear",) * (n_layers % 4)
            if len(layer_types) != n_layers:
                raise ValueError(
                    f"layer_types length {len(layer_types)} != "
                    f"num_hidden_layers {n_layers}"
                )

            return cls(
                arch="qwen3_5_hybrid",
                vocab_size=cfg["vocab_size"],
                hidden_size=hidden,
                intermediate_size=cfg["intermediate_size"],
                num_attention_heads=n_heads,
                num_key_value_heads=cfg.get("num_key_value_heads", n_heads),
                num_hidden_layers=n_layers,
                rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
                rope_theta=cfg.get("rope_theta", 10000000.0),
                head_dim=head_dim,
                tie_word_embeddings=cfg.get("tie_word_embeddings", False),
                max_position_embeddings=cfg.get("max_position_embeddings", 262144),
                fused_qkv=False,
                fused_gate_up=False,
                has_qkv_bias=False,  # Qwen3.5 full-attention layers have no QKV bias
                layer_types=layer_types,
                partial_rotary_factor=cfg.get("partial_rotary_factor", 0.25),
                attn_output_gate=cfg.get("attn_output_gate", True),
                linear_num_k_heads=cfg.get("linear_num_key_heads", 16),
                linear_num_v_heads=cfg.get("linear_num_value_heads", 48),
                linear_head_k_dim=cfg.get("linear_key_head_dim", 128),
                linear_head_v_dim=cfg.get("linear_value_head_dim", 128),
                linear_conv_kernel_dim=cfg.get("linear_conv_kernel_dim", 4),
                norm_residual_weight=True,  # Qwen3.5 uses (1+w) RMSNorm
            )
        elif model_type == "qwen3":
            return cls(
                arch="qwen3",
                vocab_size=cfg["vocab_size"],
                hidden_size=hidden,
                intermediate_size=cfg["intermediate_size"],
                num_attention_heads=n_heads,
                num_key_value_heads=cfg.get("num_key_value_heads", n_heads),
                num_hidden_layers=cfg["num_hidden_layers"],
                rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
                rope_theta=cfg.get("rope_theta", 1000000.0),
                head_dim=head_dim,
                tie_word_embeddings=cfg.get("tie_word_embeddings", False),
                max_position_embeddings=cfg.get("max_position_embeddings", 32768),
                fused_qkv=False,
                fused_gate_up=False,
                has_qkv_bias=cfg.get("attention_bias", False),
            )
        elif model_type == "qwen2":
            return cls(
                arch="qwen2",
                vocab_size=cfg["vocab_size"],
                hidden_size=hidden,
                intermediate_size=cfg["intermediate_size"],
                num_attention_heads=n_heads,
                num_key_value_heads=cfg.get("num_key_value_heads", n_heads),
                num_hidden_layers=cfg["num_hidden_layers"],
                rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
                rope_theta=cfg.get("rope_theta", 10000.0),
                head_dim=head_dim,
                tie_word_embeddings=cfg.get("tie_word_embeddings", False),
                max_position_embeddings=cfg.get("max_position_embeddings", 32768),
                fused_qkv=False,
                fused_gate_up=False,
                has_qkv_bias=True,
            )
        elif model_type == "phi3":
            rope_scaling = cfg.get("rope_scaling", {})
            rope_type = (rope_scaling.get("rope_type") or rope_scaling.get("type")) if rope_scaling else None
            long_factor = tuple(rope_scaling["long_factor"]) if rope_scaling and "long_factor" in rope_scaling else None
            short_factor = tuple(rope_scaling["short_factor"]) if rope_scaling and "short_factor" in rope_scaling else None
            orig_max_pos = cfg.get("original_max_position_embeddings", 0)
            return cls(
                arch="phi3",
                vocab_size=cfg["vocab_size"],
                hidden_size=hidden,
                intermediate_size=cfg["intermediate_size"],
                num_attention_heads=n_heads,
                num_key_value_heads=cfg.get("num_key_value_heads", n_heads),
                num_hidden_layers=cfg["num_hidden_layers"],
                rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
                rope_theta=cfg.get("rope_theta", 10000.0),
                head_dim=head_dim,
                tie_word_embeddings=cfg.get("tie_word_embeddings", False),
                max_position_embeddings=cfg.get("max_position_embeddings", 131072),
                fused_qkv=True,
                fused_gate_up=True,
                has_qkv_bias=False,
                partial_rotary_factor=cfg.get("partial_rotary_factor", 1.0),
                rope_scaling_type=rope_type,
                rope_long_factor=long_factor,
                rope_short_factor=short_factor,
                original_max_position_embeddings=orig_max_pos,
            )
        elif model_type in ("llama", "mistral", "ministral"):
            # Parse rope_scaling for llama3 / linear / etc.
            # Mistral3 commonly uses rope_parameters with the same keys.
            rope_scaling = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
            rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
            rs_factor = float(rope_scaling.get("factor", 1.0))
            rs_low_freq = float(rope_scaling.get("low_freq_factor", 1.0))
            rs_high_freq = float(rope_scaling.get("high_freq_factor", 1.0))
            rs_orig_max_pos = int(rope_scaling.get("original_max_position_embeddings", 0))
            return cls(
                arch="llama",
                vocab_size=cfg["vocab_size"],
                hidden_size=hidden,
                intermediate_size=cfg["intermediate_size"],
                num_attention_heads=n_heads,
                num_key_value_heads=cfg.get("num_key_value_heads", n_heads),
                num_hidden_layers=cfg["num_hidden_layers"],
                rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
                rope_theta=cfg.get("rope_theta", rope_scaling.get("rope_theta", 500000.0)),
                head_dim=head_dim,
                tie_word_embeddings=cfg.get("tie_word_embeddings", True),
                max_position_embeddings=cfg.get("max_position_embeddings", 131072),
                fused_qkv=False,
                fused_gate_up=False,
                has_qkv_bias=False,
                rope_scaling_type=rope_type,
                original_max_position_embeddings=rs_orig_max_pos,
                rope_scaling_factor=rs_factor,
                rope_scaling_low_freq_factor=rs_low_freq,
                rope_scaling_high_freq_factor=rs_high_freq,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")


# ─── Weight loading ───────────────────────────────────────────────────────────

def _load_all_safetensors(model_dir: Path) -> dict[str, Any]:
    """Load all safetensors shards from a model directory.

    Returns numpy or JAX arrays depending on weight dtype.
    Models with bf16 weights (e.g. Mistral) are loaded via the flax framework
    since numpy does not support bfloat16; other models use numpy directly.
    Skips consolidated.safetensors (Mistral native format with different keys).
    """
    shards = sorted(model_dir.glob("model*.safetensors"))
    if not shards:
        # Fallback: try any .safetensors except consolidated
        shards = sorted(
            s for s in model_dir.glob("*.safetensors")
            if "consolidated" not in s.name
        )
    if not shards:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")

    # Try numpy first (works for float16/float32); fall back to flax for bf16
    params: dict[str, Any] = {}
    n_shards = len(shards)
    try:
        from safetensors.numpy import load_file
        for i, shard in enumerate(shards, 1):
            print(f"  [shard {i}/{n_shards}] {shard.name}", flush=True)
            params.update(load_file(str(shard)))
    except (TypeError, AttributeError):
        # bf16/float8 weights or older numpy dtype support gaps.
        # Load as JAX arrays directly via flax path.
        from safetensors import safe_open
        params = {}
        for i, shard in enumerate(shards, 1):
            print(f"  [shard {i}/{n_shards}] {shard.name}", flush=True)
            with safe_open(str(shard), framework="flax") as f:
                for key in f.keys():
                    params[key] = f.get_tensor(key)
    print(f"  Loaded {len(params)} tensors from {n_shards} shards", flush=True)
    return params


def load_model(model_dir: str | Path) -> tuple[TransformerConfig, dict[str, Any]]:
    """Load config + weights from a HuggingFace model directory.

    Returns (config, params) where params are JAX bf16 arrays on the
    default device.  For VLMs (e.g. Qwen3.5), only text-model weights
    (model.* and lm_head.*) are loaded; vision-encoder weights are skipped.
    """
    model_dir = Path(model_dir)
    config = TransformerConfig.from_hf_config(model_dir / "config.json")

    raw_params = _load_all_safetensors(model_dir)

    # Normalize VLM-style key prefixes:
    #   model.language_model.layers.X.* → model.layers.X.*
    #   model.language_model.embed_tokens.* → model.embed_tokens.*
    # Also strip model.visual.* (vision encoder)
    has_lm_prefix = any(k.startswith("model.language_model.") for k in raw_params)
    if has_lm_prefix:
        normalized: dict[str, Any] = {}
        n_vision = 0
        for k, v in raw_params.items():
            if k.startswith("model.language_model."):
                new_key = "model." + k[len("model.language_model."):]
                normalized[new_key] = v
            elif k.startswith("model.visual."):
                n_vision += 1
            elif k.startswith("lm_head."):
                normalized[k] = v
            # Skip mtp.* and other non-text params
        if n_vision > 0:
            print(f"[JAX] Skipped {n_vision} vision encoder params")
        del raw_params
        raw_params = normalized

    # Normalize Gemma3-style key prefixes:
    #   language_model.model.* → model.*
    has_gemma3_prefix = any(k.startswith("language_model.model.") for k in raw_params)
    if has_gemma3_prefix:
        normalized = {}
        n_skipped_g3 = 0
        for k, v in raw_params.items():
            if k.startswith("language_model.model."):
                new_key = "model." + k[len("language_model.model."):]
                normalized[new_key] = v
            elif k.startswith("language_model.lm_head."):
                new_key = k[len("language_model."):]  # → "lm_head.weight"
                normalized[new_key] = v
            elif k.startswith(("vision_tower.", "multi_modal_projector.")):
                n_skipped_g3 += 1
            elif k.startswith("lm_head."):
                normalized[k] = v
            else:
                n_skipped_g3 += 1
        if n_skipped_g3 > 0:
            print(f"[JAX] Skipped {n_skipped_g3} non-text params (vision/projector)")
        del raw_params
        raw_params = normalized

    # Filter to text-model weights only (skip vision encoder for VLMs)
    text_prefixes = ("model.", "lm_head.")
    text_params = {k: v for k, v in raw_params.items()
                   if any(k.startswith(p) for p in text_prefixes)}
    n_skipped = len(raw_params) - len(text_params)
    if n_skipped > 0:
        print(f"[JAX] Skipped {n_skipped} non-text params")
    del raw_params
    # Convert to bf16 JAX arrays
    params = {k: jnp.asarray(v, dtype=jnp.bfloat16) for k, v in text_params.items()}
    del text_params
    return config, params


# ─── Tensor parallelism ──────────────────────────────────────────────────────

def create_tp_mesh(tp_size: int | None = None) -> Mesh | None:
    """Create a 1D tensor-parallel mesh.

    Preference order:
     1) Local-host mesh when tp_size fits on this host.
         This avoids multi-host device_put validation traffic for param loading.
     2) Balanced multi-host global mesh when tp_size exceeds local devices and
         can be split evenly across processes.

    Returns None if tp_size is None or <= 1.
    """
    if tp_size is None or tp_size <= 1:
        return None

    local_devices = list(jax.local_devices())
    global_devices = list(jax.devices())
    process_count = jax.process_count()

    # Prefer local mesh whenever possible. In distributed runs this keeps
    # sharding fully addressable on each host and avoids multi-host allgather
    # during device_put for large parameter tensors.
    if len(local_devices) >= tp_size:
        return Mesh(np.array(local_devices[:tp_size]), axis_names=("tp",))

    if process_count > 1 and tp_size % process_count == 0:
        per_process = tp_size // process_count
        if per_process > 0 and len(local_devices) >= per_process:
            by_process: dict[int, list[Any]] = {}
            for d in global_devices:
                by_process.setdefault(int(d.process_index), []).append(d)

            selected: list[Any] = []
            ok = True
            for p in range(process_count):
                proc_devs = sorted(by_process.get(p, []), key=lambda d: int(d.id))
                if len(proc_devs) < per_process:
                    ok = False
                    break
                selected.extend(proc_devs[:per_process])

            if ok and len(selected) == tp_size:
                return Mesh(np.array(selected), axis_names=("tp",))

    raise ValueError(
        "Requested tp_size="
        f"{tp_size} but only {len(local_devices)} local devices are available; "
        "multi-host balanced mesh setup was not possible"
    )


def _shard_weight(
    param: Any, name: str, config: TransformerConfig, mesh: Mesh,
) -> Any:
    """Apply tensor-parallel sharding to a single weight tensor.

    Column-parallel (shard axis 0): Q/K/V/gate/up projections + biases
    Row-parallel (shard axis 1): O/down projections
    Replicated: embeddings, norms, lm_head, DeltaNet scalars
    """
    ndim = int(getattr(param, "ndim", 0))

    # Some checkpoints include scalar auxiliary tensors (e.g. per-tensor scales).
    # These should always be replicated.
    if ndim == 0:
        return jax.device_put(param, NamedSharding(mesh, P()))

    # Determine sharding based on parameter name suffix
    if any(s in name for s in (
        "q_proj.weight", "k_proj.weight", "v_proj.weight",
        "qkv_proj.weight", "gate_proj.weight", "up_proj.weight",
        "gate_up_proj.weight", "g_proj.weight",
        # DeltaNet: column-parallel projections
        "in_proj_qkv.weight", "in_proj_z.weight",
    )):
        # Column-parallel: shard output dimension (axis 0)
        spec = P("tp", None) if ndim >= 2 else P("tp")
    elif any(s in name for s in (
        "q_proj.bias", "k_proj.bias", "v_proj.bias",
    )):
        # Bias for column-parallel projections
        spec = P("tp") if ndim == 1 else P(*([None] * ndim))
    elif any(s in name for s in (
        "o_proj.weight", "down_proj.weight",
        # DeltaNet: row-parallel output projection
        "out_proj.weight",
    )):
        # Row-parallel: shard reduction dimension (axis 1)
        spec = P(None, "tp") if ndim >= 2 else P(*([None] * ndim))
    else:
        # Everything else is replicated: embed, norms, lm_head,
        # DeltaNet small tensors (A_log, dt_bias, conv1d, in_proj_a, in_proj_b)
        spec = P(*([None] * ndim))

    return jax.device_put(param, NamedSharding(mesh, spec))


def shard_model_params(
    params: dict[str, Any], config: TransformerConfig, mesh: Mesh,
) -> dict[str, Any]:
    """Shard all model parameters across a tensor-parallel mesh."""
    sharded = {}
    for name, param in params.items():
        sharded[name] = _shard_weight(param, name, config, mesh)
    return sharded


# ─── Primitive ops ────────────────────────────────────────────────────────────

def rms_norm(x: Any, weight: Any, eps: float, residual_weight: bool = True) -> Any:
    """RMSNorm.  Qwen3.5 layernorms use residual_weight=True: y = x*(1+w)/rms.
    The DeltaNet gated norm uses residual_weight=False: y = x*w/rms."""
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    normed = x_f32 * jax.lax.rsqrt(variance + eps)
    w = weight.astype(jnp.float32)
    if residual_weight:
        w = 1.0 + w
    return (normed * w).astype(x.dtype)


def rotary_embedding(x: Any, positions: Any, head_dim: int, theta: float) -> Any:
    """Apply RoPE to x. x shape: (batch, n_heads, seq_len, head_dim)."""
    half = head_dim // 2
    freq_exponents = jnp.arange(0, half, dtype=jnp.float32) / half
    inv_freq = 1.0 / (theta ** freq_exponents)  # (half,)

    # positions: (seq_len,) -> (1, 1, seq_len, 1)
    pos = positions.astype(jnp.float32)
    angles = pos[:, None] * inv_freq[None, :]  # (seq_len, half)
    angles = angles[None, None, :, :]  # (1, 1, seq_len, half)

    cos = jnp.cos(angles).astype(x.dtype)
    sin = jnp.sin(angles).astype(x.dtype)

    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def longrope_rotary_embedding(
    x: Any, positions: Any, head_dim: int, theta: float,
    long_factor: tuple[float, ...], short_factor: tuple[float, ...],
    original_max_position_embeddings: int, max_position_embeddings: int,
    partial_rotary_factor: float = 1.0,
) -> Any:
    """Apply LongRoPE (SuRoPE) to x. x shape: (batch, n_heads, seq_len, head_dim).

    Phi-3.5 uses per-dimension frequency scaling factors that differ for
    short (≤ original_max_pos) and long (> original_max_pos) sequences.
    The factor divides inv_freq, effectively stretching the period.

    When partial_rotary_factor < 1.0 (e.g. Phi-4-mini at 0.75), only the
    first rotary_dim dimensions get RoPE; the rest pass through unchanged.
    The long/short factors have length rotary_dim/2, not head_dim/2.
    """
    rot_dim = int(head_dim * partial_rotary_factor)
    rot_dim = rot_dim - (rot_dim % 2)  # ensure even
    half = rot_dim // 2
    freq_exponents = jnp.arange(0, half, dtype=jnp.float32) / half
    inv_freq = 1.0 / (theta ** freq_exponents)  # (half,)

    # Select long or short factor based on max position in the sequence.
    # At inference time with a KV cache, positions may span the whole context,
    # so we use max_position_embeddings > original_max_position_embeddings as proxy.
    if max_position_embeddings > original_max_position_embeddings:
        factor = jnp.array(long_factor, dtype=jnp.float32)
    else:
        factor = jnp.array(short_factor, dtype=jnp.float32)

    inv_freq = inv_freq / factor  # (half,) — per-dimension scaling

    pos = positions.astype(jnp.float32)
    angles = pos[:, None] * inv_freq[None, :]  # (seq_len, half)
    angles = angles[None, None, :, :]  # (1, 1, seq_len, half)

    cos = jnp.cos(angles).astype(x.dtype)
    sin = jnp.sin(angles).astype(x.dtype)

    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    x_rot = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    if partial_rotary_factor >= 1.0:
        return x_rot
    return jnp.concatenate([x_rot, x_pass], axis=-1)


def llama3_rotary_embedding(
    x: Any, positions: Any, head_dim: int, theta: float,
    factor: float, low_freq_factor: float, high_freq_factor: float,
    original_max_position_embeddings: int,
) -> Any:
    """Apply Llama-3.1 RoPE scaling. x shape: (batch, n_heads, seq_len, head_dim).

    Uses piecewise frequency scaling:
    - High-frequency dims (short wavelength < old_ctx/high_freq_factor): keep original
    - Low-frequency dims (long wavelength > old_ctx/low_freq_factor): scale by 1/factor
    - Mid-frequency dims: smooth interpolation between scaled and unscaled
    """
    half = head_dim // 2
    freq_exponents = jnp.arange(0, half, dtype=jnp.float32) / half
    inv_freq = 1.0 / (theta ** freq_exponents)  # (half,)

    old_ctx = float(original_max_position_embeddings)
    low_freq_wavelen = old_ctx / low_freq_factor
    high_freq_wavelen = old_ctx / high_freq_factor

    wavelen = 2.0 * jnp.pi / inv_freq  # (half,)

    # Piecewise: high freq (wavelen < high_freq_wavelen) → keep
    #            low freq (wavelen > low_freq_wavelen) → scale by 1/factor
    #            mid freq → smooth interpolation
    smooth = (old_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smooth = jnp.clip(smooth, 0.0, 1.0)

    scaled_inv_freq = inv_freq / factor
    inv_freq = smooth * inv_freq + (1.0 - smooth) * scaled_inv_freq  # (half,)

    pos = positions.astype(jnp.float32)
    angles = pos[:, None] * inv_freq[None, :]  # (seq_len, half)
    angles = angles[None, None, :, :]  # (1, 1, seq_len, half)

    cos = jnp.cos(angles).astype(x.dtype)
    sin = jnp.sin(angles).astype(x.dtype)

    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def silu(x: Any) -> Any:
    """SiLU / swish activation."""
    return x * jax.nn.sigmoid(x)


def _l2_normalize(x: Any, axis: int = -1) -> Any:
    """L2-normalize along an axis (for DeltaNet Q/K normalization)."""
    norm = jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=True) + 1e-12)
    return x / norm


def rotary_embedding_partial(
    x: Any, positions: Any, head_dim: int, theta: float, partial_factor: float,
) -> Any:
    """Apply RoPE to only the first partial_factor fraction of head_dim.

    Used by Qwen3.5 full-attention layers (partial_rotary_factor=0.25).
    """
    if partial_factor >= 1.0:
        return rotary_embedding(x, positions, head_dim, theta)
    rot_dim = int(head_dim * partial_factor)
    rot_dim = rot_dim - (rot_dim % 2)  # ensure even
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]
    x_rot = rotary_embedding(x_rot, positions, rot_dim, theta)
    return jnp.concatenate([x_rot, x_pass], axis=-1)


# ─── Gated DeltaNet linear attention ──────────────────────────────────────────

def _deltanet_layer(
    hidden: Any,
    layer_params: dict[str, Any],
    recurrent_state: Any,
    conv_state: Any,
    config: TransformerConfig,
) -> tuple[Any, Any, Any]:
    """Single Gated DeltaNet linear attention layer (supports any seq_len).

    Args:
        hidden: (B, seq_len, hidden_size)
        layer_params: dict with DeltaNet weight tensors
        recurrent_state: (B, num_v_heads, head_k_dim, head_v_dim) in bf16
        conv_state: (B, key_dim, conv_kernel_dim)
        config: TransformerConfig with linear attention settings

    Returns: (output, new_recurrent_state, new_conv_state)
    """
    num_k_heads = config.linear_num_k_heads
    num_v_heads = config.linear_num_v_heads
    head_k_dim = config.linear_head_k_dim
    head_v_dim = config.linear_head_v_dim
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    v_per_k = num_v_heads // num_k_heads  # heads ratio (3 for Qwen3.5)

    batch, seq_len, _ = hidden.shape

    # 1. Batch projections across all tokens
    h_flat = hidden.reshape(batch * seq_len, -1)  # (B*S, hidden)
    qkv_all = h_flat @ layer_params["in_proj_qkv.weight"].T
    z_all = h_flat @ layer_params["in_proj_z.weight"].T
    beta_raw_all = h_flat @ layer_params["in_proj_b.weight"].T
    alpha_raw_all = h_flat @ layer_params["in_proj_a.weight"].T

    # Reshape to (B, S, dim)
    qkv_all = qkv_all.reshape(batch, seq_len, -1)
    z_all = z_all.reshape(batch, seq_len, value_dim)
    beta_raw_all = beta_raw_all.reshape(batch, seq_len, num_v_heads)
    alpha_raw_all = alpha_raw_all.reshape(batch, seq_len, num_v_heads)

    # Pre-compute conv/gating constants
    # conv1d operates on full QKV (dim = key_dim*2 + value_dim = 10240 for Qwen3.5)
    conv_w = layer_params["conv1d.weight"][:, 0, :]  # (qkv_dim, kernel_dim)
    conv_b = layer_params.get("conv1d.bias")  # optional
    A_log = layer_params["A_log"]
    dt_bias = layer_params["dt_bias"]
    attn_norm_w = layer_params.get("norm.weight")  # internal norm after recurrent readout

    # 2. Sequential scan: conv1d + recurrent update per token.
    # Transpose to (S, B, dim) for scan leading axis.
    def _scan_step(carry, x):
        rec_st, c_st = carry  # rec_st: (B, v_heads, k_dim, v_dim) bf16
        qkv_t, z_t, beta_raw_t, alpha_raw_t = x

        # Conv1d on full QKV: shift state, insert new token, depthwise conv
        c_st = jnp.concatenate([c_st[:, :, 1:], qkv_t[:, :, None]], axis=2)
        qkv_conv = jnp.sum(c_st * conv_w[None, :, :], axis=-1)  # (B, qkv_dim)
        if conv_b is not None:
            qkv_conv = qkv_conv + conv_b

        # Apply silu to entire QKV (matches HF causal_conv1d_fn activation),
        # then split into Q, K, V
        qkv_conv = silu(qkv_conv)
        q_t = qkv_conv[:, :key_dim]
        k_t = qkv_conv[:, key_dim:key_dim * 2]
        v_t = qkv_conv[:, key_dim * 2:]

        # Reshape to heads
        q_t = q_t.reshape(batch, num_k_heads, head_k_dim)
        k_t = k_t.reshape(batch, num_k_heads, head_k_dim)
        v_t = v_t.reshape(batch, num_v_heads, head_v_dim)
        z_t = z_t.reshape(batch, num_v_heads, head_v_dim)

        # Gating
        beta = jax.nn.sigmoid(beta_raw_t)  # (B, v_heads)
        g = -jnp.exp(A_log)[None, :] * jax.nn.softplus(
            alpha_raw_t + dt_bias[None, :])  # (B, v_heads)

        # Repeat K heads to match V heads
        q_t = jnp.repeat(q_t, v_per_k, axis=1)
        k_t = jnp.repeat(k_t, v_per_k, axis=1)

        # L2-normalize Q and K, then scale Q by 1/sqrt(head_k_dim)
        # (matches HF torch_recurrent_gated_delta_rule)
        q_t = _l2_normalize(q_t, axis=-1)
        k_t = _l2_normalize(k_t, axis=-1)
        q_t = q_t * (1.0 / (head_k_dim ** 0.5))

        # Recurrent update in float32 for numerical stability
        rec_f = rec_st.astype(jnp.float32)
        q_f = q_t.astype(jnp.float32)
        k_f = k_t.astype(jnp.float32)
        v_f = v_t.astype(jnp.float32)

        g_exp = jnp.exp(g.astype(jnp.float32))[:, :, None, None]
        rec_f = rec_f * g_exp

        kv_mem = jnp.sum(rec_f * k_f[:, :, :, None], axis=-2)
        delta = (v_f - kv_mem) * beta.astype(jnp.float32)[:, :, None]
        rec_f = rec_f + k_f[:, :, :, None] * delta[:, :, None, :]

        attn_out_t = jnp.sum(rec_f * q_f[:, :, :, None], axis=-2)  # (B, v_heads, v_dim)
        attn_out_t = attn_out_t.astype(jnp.bfloat16)

        # Apply internal norm if present (stabilizes recurrent output)
        # norm.weight is (head_v_dim,) — apply per-head on last axis
        if attn_norm_w is not None:
            attn_out_t = rms_norm(attn_out_t, attn_norm_w, 1e-6, residual_weight=False)

        attn_out_t = (attn_out_t * silu(z_t))

        rec_st = rec_f.astype(jnp.bfloat16)
        return (rec_st, c_st), attn_out_t  # attn_out_t: (B, v_heads, v_dim)

    # Pack inputs: leading axis = S (sequence)
    xs = (
        jnp.moveaxis(qkv_all, 1, 0),       # (S, B, qkv_dim)
        jnp.moveaxis(z_all, 1, 0),         # (S, B, value_dim)
        jnp.moveaxis(beta_raw_all, 1, 0),  # (S, B, v_heads)
        jnp.moveaxis(alpha_raw_all, 1, 0), # (S, B, v_heads)
    )

    (recurrent_state, conv_state), outputs = jax.lax.scan(
        _scan_step, (recurrent_state, conv_state), xs)
    # outputs: (S, B, v_heads, v_dim)

    # 3. Reshape and output projection
    attn_out = jnp.moveaxis(outputs, 0, 1)  # (B, S, v_heads, v_dim)
    attn_out = attn_out.reshape(batch, seq_len, value_dim)
    output = attn_out @ layer_params["out_proj.weight"].T  # (B, S, hidden_size)

    return output, recurrent_state, conv_state


# ─── Attention ────────────────────────────────────────────────────────────────

# (KVCache replaced by flat cache_keys/cache_values lists for JIT compatibility)


def _attention_layer(
    hidden: Any,
    layer_params: dict[str, Any],
    positions: Any,
    cache_k: Any,
    cache_v: Any,
    cache_pos: Any,
    mask: Any,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rope_theta: float,
    fused_qkv: bool,
    has_qkv_bias: bool,
    output_attentions: bool,
    partial_rotary_factor: float = 1.0,
    output_gate: bool = False,
    rope_scaling_type: str | None = None,
    rope_long_factor: tuple[float, ...] | None = None,
    rope_short_factor: tuple[float, ...] | None = None,
    original_max_position_embeddings: int = 0,
    max_position_embeddings: int = 0,
    rope_scaling_factor: float = 1.0,
    rope_scaling_low_freq_factor: float = 1.0,
    rope_scaling_high_freq_factor: float = 1.0,
    norm_residual_weight: bool = False,
    query_pre_attn_scalar: float = 0,
    rms_norm_eps: float = 1e-6,
    sliding_window: int | None = None,
) -> tuple[Any, Any, Any, Any | None]:
    """Single-layer attention with static-shape KV cache.

    cache_k/v: (B, n_kv_heads, max_cache_len, head_dim) - full static-shape cache.
    cache_pos: scalar int32 - current write position in cache.
    mask: (1, 1, 1, max_cache_len) - additive mask (-1e9 for blocked positions).
    Returns: (output, new_cache_k, new_cache_v, attn_weights_or_None).
    """
    n_groups = n_heads // n_kv_heads
    batch, seq_len, _ = hidden.shape

    if fused_qkv:
        qkv = hidden @ layer_params["qkv_proj.weight"].T
        q_dim = n_heads * head_dim
        k_dim = n_kv_heads * head_dim
        q, k, v = qkv[..., :q_dim], qkv[..., q_dim:q_dim + k_dim], qkv[..., q_dim + k_dim:]
        gate_raw = None
    else:
        q_raw = hidden @ layer_params["q_proj.weight"].T
        k = hidden @ layer_params["k_proj.weight"].T
        v = hidden @ layer_params["v_proj.weight"].T
        if has_qkv_bias:
            q_raw = q_raw + layer_params["q_proj.bias"]
            k = k + layer_params["k_proj.bias"]
            v = v + layer_params["v_proj.bias"]

        # Qwen3.5: q_proj fuses [Q; gate] when output_gate=True and no separate g_proj
        # HF layout: view as (B, S, n_heads, head_dim*2) then chunk → per-head interleaved
        q_dim = n_heads * head_dim
        if output_gate and "g_proj.weight" not in layer_params and q_raw.shape[-1] == 2 * q_dim:
            q_gate = q_raw.reshape(batch, seq_len, n_heads, head_dim * 2)
            q = q_gate[..., :head_dim].reshape(batch, seq_len, q_dim)
            gate_raw = q_gate[..., head_dim:].reshape(batch, seq_len, q_dim)
        else:
            q = q_raw
            gate_raw = None

    q = q.reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

    # QK norms (Qwen3, Gemma3): per-head RMS norm on Q and K before RoPE
    # q_norm.weight / k_norm.weight are (head_dim,) — broadcast on last axis naturally
    # Gemma3 uses (1+w) RMSNorm; Qwen3 uses standard RMSNorm
    if "q_norm.weight" in layer_params:
        q = rms_norm(q, layer_params["q_norm.weight"], rms_norm_eps, residual_weight=norm_residual_weight)
    if "k_norm.weight" in layer_params:
        k = rms_norm(k, layer_params["k_norm.weight"], rms_norm_eps, residual_weight=norm_residual_weight)

    if rope_scaling_type == "longrope" and rope_long_factor is not None:
        q = longrope_rotary_embedding(
            q, positions, head_dim, rope_theta,
            rope_long_factor, rope_short_factor,
            original_max_position_embeddings, max_position_embeddings,
            partial_rotary_factor,
        )
        k = longrope_rotary_embedding(
            k, positions, head_dim, rope_theta,
            rope_long_factor, rope_short_factor,
            original_max_position_embeddings, max_position_embeddings,
            partial_rotary_factor,
        )
    elif rope_scaling_type == "llama3" and original_max_position_embeddings > 0:
        q = llama3_rotary_embedding(
            q, positions, head_dim, rope_theta,
            rope_scaling_factor, rope_scaling_low_freq_factor,
            rope_scaling_high_freq_factor, original_max_position_embeddings,
        )
        k = llama3_rotary_embedding(
            k, positions, head_dim, rope_theta,
            rope_scaling_factor, rope_scaling_low_freq_factor,
            rope_scaling_high_freq_factor, original_max_position_embeddings,
        )
    else:
        q = rotary_embedding_partial(q, positions, head_dim, rope_theta, partial_rotary_factor)
        k = rotary_embedding_partial(k, positions, head_dim, rope_theta, partial_rotary_factor)

    # Write new K/V into static-shape cache using dynamic_update_slice
    # cache shape: (B, n_kv_heads, max_cache_len, head_dim)
    # k shape: (B, n_kv_heads, seq_len, head_dim)
    cache_k = jax.lax.dynamic_update_slice(cache_k, k, (0, 0, cache_pos, 0))
    cache_v = jax.lax.dynamic_update_slice(cache_v, v, (0, 0, cache_pos, 0))

    # Attend over full cache (mask blocks unfilled positions)
    scale = 1.0 / math.sqrt(query_pre_attn_scalar if query_pre_attn_scalar > 0 else head_dim)
    cache_len = cache_k.shape[2]

    if n_groups > 1:
        # Grouped query attention — avoid materializing 8× expanded KV.
        # Reshape Q: (B, n_heads, S, D) → (B, n_kv_heads, n_groups, S, D)
        q_grouped = q.reshape(batch, n_kv_heads, n_groups, seq_len, head_dim)
        # QK^T per group, reading K at native shape
        attn_weights = jnp.einsum(
            'bkgsd,bkcd->bkgsc', q_grouped, cache_k) * scale
        # Flatten groups → (B, n_heads, S, C) for mask + softmax
        attn_weights = attn_weights.reshape(batch, n_heads, seq_len, cache_len)
        if sliding_window is not None and sliding_window > 0:
            key_pos = jnp.arange(cache_len, dtype=jnp.int32)
            if seq_len == 1:
                # Decode: current query index equals cache_pos.
                q_pos = jnp.asarray(cache_pos, dtype=jnp.int32)
                sw_valid = key_pos >= (q_pos - sliding_window + 1)
                sw_mask = jnp.where(sw_valid, 0.0, -1e9).astype(jnp.float32)
                attn_weights = attn_weights + sw_mask[None, None, None, :]
            else:
                # Prefill: query indices are [cache_pos, ..., cache_pos + seq_len - 1].
                q_pos = jnp.arange(seq_len, dtype=jnp.int32) + jnp.asarray(cache_pos, dtype=jnp.int32)
                sw_valid = key_pos[None, :] >= (q_pos[:, None] - sliding_window + 1)
                sw_mask = jnp.where(sw_valid, 0.0, -1e9).astype(jnp.float32)
                attn_weights = attn_weights + sw_mask[None, None, :, :]
        attn_weights = attn_weights + mask
        attn_weights_f32 = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1)
        # V matmul in grouped form (keep float32 V for numerical stability)
        aw_grouped = attn_weights_f32.reshape(
            batch, n_kv_heads, n_groups, seq_len, cache_len)
        attn_out = jnp.einsum(
            'bkgsc,bkcd->bkgsd', aw_grouped,
            cache_v.astype(jnp.float32))
        attn_out = attn_out.reshape(
            batch, n_heads, seq_len, head_dim).astype(jnp.bfloat16)
    else:
        attn_weights = (q @ cache_k.transpose(0, 1, 3, 2)) * scale
        if sliding_window is not None and sliding_window > 0:
            key_pos = jnp.arange(cache_len, dtype=jnp.int32)
            if seq_len == 1:
                q_pos = jnp.asarray(cache_pos, dtype=jnp.int32)
                sw_valid = key_pos >= (q_pos - sliding_window + 1)
                sw_mask = jnp.where(sw_valid, 0.0, -1e9).astype(jnp.float32)
                attn_weights = attn_weights + sw_mask[None, None, None, :]
            else:
                q_pos = jnp.arange(seq_len, dtype=jnp.int32) + jnp.asarray(cache_pos, dtype=jnp.int32)
                sw_valid = key_pos[None, :] >= (q_pos[:, None] - sliding_window + 1)
                sw_mask = jnp.where(sw_valid, 0.0, -1e9).astype(jnp.float32)
                attn_weights = attn_weights + sw_mask[None, None, :, :]
        attn_weights = attn_weights + mask
        attn_weights_f32 = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1)
        # Keep float32 through the V matmul to preserve long-tail attention weights
        attn_out = (attn_weights_f32 @ cache_v.astype(jnp.float32)).astype(jnp.bfloat16)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)

    # Output gating (Qwen3.5 full-attention layers): gate applied before o_proj
    # HF uses sigmoid (not silu) for the attention output gate
    if output_gate:
        if gate_raw is not None:
            # Gate was fused into q_proj: gate_raw shape (B, S, n_heads * head_dim)
            attn_out = attn_out * jax.nn.sigmoid(gate_raw.astype(jnp.float32)).astype(jnp.bfloat16)
        elif "g_proj.weight" in layer_params:
            gate = jax.nn.sigmoid((hidden @ layer_params["g_proj.weight"].T).astype(jnp.float32)).astype(jnp.bfloat16)
            attn_out = attn_out * gate

    output = attn_out @ layer_params["o_proj.weight"].T

    scores = attn_weights_f32 if output_attentions else None
    return output, cache_k, cache_v, scores


# ─── MLP ──────────────────────────────────────────────────────────────────────

def _mlp_layer(
    hidden: Any,
    layer_params: dict[str, Any],
    fused_gate_up: bool,
    hidden_activation: str = "silu",
) -> Any:
    """Single-layer MLP forward pass."""
    if fused_gate_up:
        gate_up = hidden @ layer_params["gate_up_proj.weight"].T
        gate, up = jnp.split(gate_up, 2, axis=-1)
    else:
        gate = hidden @ layer_params["gate_proj.weight"].T
        up = hidden @ layer_params["up_proj.weight"].T

    if hidden_activation == "gelu_tanh":
        activated = jax.nn.gelu(gate, approximate=True)
    else:
        activated = silu(gate)
    return (activated * up) @ layer_params["down_proj.weight"].T


# ─── Full transformer forward ────────────────────────────────────────────────

def _organize_layer_params(params: dict[str, Any], config: TransformerConfig) -> list[dict[str, Any]]:
    """Reorganize flat params dict into per-layer dicts for efficient access.

    Handles both standard attention and DeltaNet linear attention weight names.
    For hybrid models (Qwen3.5), DeltaNet layers use `linear_attn.` prefix
    while full attention layers use `self_attn.` prefix.
    """
    layers = []
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}"
        layer: dict[str, Any] = {}

        # Determine attention prefix based on layer type
        is_linear = (config.layer_types is not None
                     and config.layer_types[i] == "linear")
        attn_prefixes = []
        if is_linear:
            attn_prefixes.append(f"{prefix}.linear_attn.")
        attn_prefixes.append(f"{prefix}.self_attn.")

        for attn_prefix in attn_prefixes:
            for k, v in params.items():
                if k.startswith(attn_prefix):
                    layer[k[len(attn_prefix):]] = v

        # MLP params
        mlp_prefix = f"{prefix}.mlp."
        for k, v in params.items():
            if k.startswith(mlp_prefix):
                layer[k[len(mlp_prefix):]] = v
        # Norm params
        layer["input_layernorm.weight"] = params[f"{prefix}.input_layernorm.weight"]
        layer["post_attention_layernorm.weight"] = params[f"{prefix}.post_attention_layernorm.weight"]
        # Gemma3 sandwich norms
        ffn_pre_key = f"{prefix}.pre_feedforward_layernorm.weight"
        if ffn_pre_key in params:
            layer["pre_feedforward_layernorm.weight"] = params[ffn_pre_key]
        ffn_post_key = f"{prefix}.post_feedforward_layernorm.weight"
        if ffn_post_key in params:
            layer["post_feedforward_layernorm.weight"] = params[ffn_post_key]
        layers.append(layer)
    return layers


def _build_position_mask(
    cache_pos: Any, seq_len: int, max_cache_len: int
) -> Any:
    """Build additive mask for static-shape cache.

    Unfilled positions (beyond cache_pos + seq_len) are masked with -1e9.
    For decode (seq_len=1): allows attending to positions 0..cache_pos.
    For prefill: causal mask over 0..seq_len-1.
    """
    positions = jnp.arange(max_cache_len, dtype=jnp.int32)  # (max_cache,)
    valid_end = cache_pos + seq_len  # positions beyond this are unfilled

    if seq_len == 1:
        # Decode: attend to all filled positions
        valid = positions < valid_end
    else:
        # Prefill: causal mask
        query_pos = jnp.arange(seq_len, dtype=jnp.int32) + cache_pos  # (seq_len,)
        valid = positions[None, :] <= query_pos[:, None]  # (seq_len, max_cache)

    mask = jnp.where(valid, 0.0, -1e9).astype(jnp.float32)
    if seq_len == 1:
        return mask[None, None, None, :]  # (1, 1, 1, max_cache)
    return mask[None, None, :, :]  # (1, 1, seq_len, max_cache)


def _forward_jittable(
    input_ids: Any,
    cache_keys: list[Any],
    cache_values: list[Any],
    cache_pos: Any,
    mask: Any,
    *,
    recurrent_states: list[Any] | None = None,
    conv_states: list[Any] | None = None,
    embed_w: Any,
    lm_head_w: Any | None,
    final_norm_w: Any,
    layer_params: list[dict[str, Any]],
    config: TransformerConfig,
    output_attentions: bool,
    per_head_attn: bool = False,
) -> tuple[Any, list[Any], list[Any], list[Any] | None, list[Any] | None, list[Any] | None]:
    """JIT-friendly forward pass with static-shape cache arrays.

    For hybrid architectures (Qwen3.5), cache_keys/cache_values are indexed only
    by GQA layer ordinal, while recurrent_states/conv_states are indexed by
    DeltaNet layer ordinal.

    Returns (logits, new_cache_keys, new_cache_values, attentions_or_None,
             new_recurrent_states_or_None, new_conv_states_or_None).
    """
    hidden = embed_w[input_ids]  # (batch, seq, hidden)
    if config.gemma_embed_scale:
        hidden = hidden * math.sqrt(config.hidden_size)

    all_attentions = [] if output_attentions else None
    new_cache_keys = []
    new_cache_values = []
    new_recurrent = [] if recurrent_states is not None else None
    new_conv = [] if conv_states is not None else None

    # Hoist position computation out of per-layer loop (invariant across layers)
    positions = jnp.arange(input_ids.shape[1], dtype=jnp.int32) + cache_pos

    # Separate counters for GQA vs DeltaNet layers
    kv_idx = 0
    dn_idx = 0
    is_hybrid = config.layer_types is not None

    for i in range(config.num_hidden_layers):
        lp = layer_params[i]
        layer_type = config.layer_types[i] if is_hybrid else "full"

        normed = rms_norm(hidden, lp["input_layernorm.weight"], config.rms_norm_eps, residual_weight=config.norm_residual_weight)

        if layer_type == "linear":
            # DeltaNet linear attention layer
            delta_params = {}
            for key in ("in_proj_qkv.weight", "in_proj_z.weight",
                        "in_proj_b.weight", "in_proj_a.weight",
                        "out_proj.weight", "conv1d.weight", "conv1d.bias",
                        "A_log", "dt_bias", "norm.weight"):
                if key in lp:
                    delta_params[key] = lp[key]

            attn_out, new_rs, new_cs = _deltanet_layer(
                normed, delta_params,
                recurrent_states[dn_idx], conv_states[dn_idx],
                config,
            )
            hidden = hidden + attn_out
            new_recurrent.append(new_rs)
            new_conv.append(new_cs)
            dn_idx += 1
        else:
            # Standard attention layer (with optional partial RoPE + output gate)
            attn_layer_params = {}
            for key in ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                         "q_proj.bias", "k_proj.bias", "v_proj.bias",
                         "o_proj.weight", "qkv_proj.weight", "g_proj.weight",
                         "q_norm.weight", "k_norm.weight"):
                if key in lp:
                    attn_layer_params[key] = lp[key]

            # Per-layer RoPE params (Gemma3 sliding/global layers use different theta)
            layer_rope_theta = config.rope_theta
            layer_positions = positions
            if config.gemma3_layer_attn_types is not None and i < len(config.gemma3_layer_attn_types):
                g3_lt = config.gemma3_layer_attn_types[i]
                if g3_lt == "global":
                    layer_rope_theta = config.gemma3_global_rope_theta
                    layer_positions = (positions.astype(jnp.float32) / config.gemma3_global_rope_factor)
                else:
                    layer_rope_theta = config.gemma3_local_rope_theta

            attn_out, ck, cv, scores = _attention_layer(
                normed, attn_layer_params,
                layer_positions,
                cache_keys[kv_idx], cache_values[kv_idx], cache_pos, mask,
                config.num_attention_heads, config.num_key_value_heads,
                config.head_dim, layer_rope_theta,
                config.fused_qkv, config.has_qkv_bias, output_attentions,
                partial_rotary_factor=config.partial_rotary_factor,
                output_gate=config.attn_output_gate,
                rope_scaling_type=config.rope_scaling_type,
                rope_long_factor=config.rope_long_factor,
                rope_short_factor=config.rope_short_factor,
                original_max_position_embeddings=config.original_max_position_embeddings,
                max_position_embeddings=config.max_position_embeddings,
                rope_scaling_factor=config.rope_scaling_factor,
                rope_scaling_low_freq_factor=config.rope_scaling_low_freq_factor,
                rope_scaling_high_freq_factor=config.rope_scaling_high_freq_factor,
                norm_residual_weight=config.norm_residual_weight,
                query_pre_attn_scalar=config.query_pre_attn_scalar,
                rms_norm_eps=config.rms_norm_eps,
                sliding_window=(
                    config.gemma3_sliding_window
                    if (
                        config.gemma3_layer_attn_types is not None
                        and i < len(config.gemma3_layer_attn_types)
                        and config.gemma3_layer_attn_types[i] == "sliding"
                    )
                    else None
                ),
            )
            if config.sandwich_norm:
                attn_out = rms_norm(attn_out, lp["post_attention_layernorm.weight"], config.rms_norm_eps, residual_weight=config.norm_residual_weight)
            hidden = hidden + attn_out
            new_cache_keys.append(ck)
            new_cache_values.append(cv)
            kv_idx += 1

            if output_attentions and scores is not None:
                all_attentions.append(scores)

        if config.sandwich_norm:
            normed2 = rms_norm(hidden, lp["pre_feedforward_layernorm.weight"], config.rms_norm_eps, residual_weight=config.norm_residual_weight)
        else:
            normed2 = rms_norm(hidden, lp["post_attention_layernorm.weight"], config.rms_norm_eps, residual_weight=config.norm_residual_weight)

        mlp_params = {}
        for key in ("gate_proj.weight", "up_proj.weight", "down_proj.weight",
                     "gate_up_proj.weight"):
            if key in lp:
                mlp_params[key] = lp[key]

        mlp_out = _mlp_layer(normed2, mlp_params, config.fused_gate_up, config.hidden_activation)
        if config.sandwich_norm:
            mlp_out = rms_norm(mlp_out, lp["post_feedforward_layernorm.weight"], config.rms_norm_eps, residual_weight=config.norm_residual_weight)
        hidden = hidden + mlp_out

    hidden = rms_norm(hidden, final_norm_w, config.rms_norm_eps, residual_weight=config.norm_residual_weight)

    if lm_head_w is not None:
        logits = hidden @ lm_head_w.T
    else:
        logits = hidden @ embed_w.T

    # Reduce heads on-device then stack → (L, B, S, C) instead of (L, B, H, S, C).
    # For MHA models (e.g. Phi-3.5, 32 heads × 32 layers), the full stack would
    # require ~1GB+ HBM and OOM on a single v5e chip.  Reducing heads first cuts
    # memory by num_heads × (32× for Phi, 2× for Qwen).
    stacked_attn = None
    if all_attentions is not None and len(all_attentions) > 0:
        if per_head_attn:
            # Keep head dimension: each layer → (B, H, S, C), stack → (L, B, H, S, C)
            stacked_attn = jnp.stack(all_attentions, axis=0)
        else:
            reduced = [jnp.max(a, axis=1) for a in all_attentions]  # (B, S, C) each
            stacked_attn = jnp.stack(reduced, axis=0)  # (L, B, S, C)

    return logits.astype(jnp.float32), new_cache_keys, new_cache_values, stacked_attn, new_recurrent, new_conv


# ─── Model adapter for KVCoupledQwen35Generator ──────────────────────────────

class _AttentionTuple:
    """Mimics the tuple-of-attention-tensors interface."""

    def __init__(self, attentions: list[Any] | None):
        self._attentions = attentions or []

    def __iter__(self):
        return iter(self._attentions)

    def __len__(self):
        return len(self._attentions)

    def __getitem__(self, idx):
        return self._attentions[idx]


class _ModelOutput:
    """Mimics HuggingFace model output."""

    def __init__(self, logits, past_key_values, attentions):
        self.logits = logits
        self.past_key_values = past_key_values
        self.attentions = attentions


# Maximum tokens per prefill chunk.  Keeps peak HBM below single-chip budget
# on TPU v5e (16 GB HBM).  For ~3B models 256 works; for ~7B models use 32.
_PREFILL_CHUNK_SIZE = 256


class JAXModelAdapter:
    """Drop-in replacement for _ModelCallAdapter used by KVCoupledQwen35Generator.

    Uses JIT-compiled forward passes with static-shape KV cache for TPU efficiency.
    Separate JIT traces for prefill (bucketed) vs decode.
    Both paths return attention scores since generators always need them.
    """

    def __init__(
        self, config: TransformerConfig, params: dict[str, Any],
        max_cache_len: int = 4096,
        eager: bool = False,
        prefill_chunk_size: int | None = None,
        mesh: Mesh | None = None,
    ):
        self.config = config
        self.params = params
        self.max_cache_len = max_cache_len
        self.eager = eager
        self.prefill_chunk_size = prefill_chunk_size or _PREFILL_CHUNK_SIZE
        self.mesh = mesh

        # Organize params for efficient layer-wise access
        self._layer_params = _organize_layer_params(params, config)
        self._embed_w = params["model.embed_tokens.weight"]
        self._final_norm_w = params["model.norm.weight"]
        self._lm_head_w = None if config.tie_word_embeddings else params.get("lm_head.weight")

        # JIT-compiled functions (created lazily)
        self._jit_decode = None
        self._jit_decode_with_attn = None
        self._jit_decode_per_head = None
        self._jit_prefill: dict[int, Any] = {}  # bucket_size -> compiled fn

    def _make_jit_fn(self, output_attentions: bool, per_head_attn: bool = False):
        """Create a JIT-compiled forward function.

        Model weights are passed as arguments (not captured in closure)
        to avoid embedding 50+GB of constants into the XLA graph.
        """
        config = self.config
        is_hybrid = config.layer_types is not None

        if is_hybrid:
            # Hybrid model: donate KV cache (args 1,2) and DeltaNet state (args 5,6)
            @partial(jax.jit, donate_argnums=(1, 2, 5, 6))
            def step_fn(token_ids, cache_keys, cache_values, cache_pos, mask,
                        recurrent_states, conv_states,
                        embed_w, lm_head_w, final_norm_w, layer_params):
                return _forward_jittable(
                    token_ids, cache_keys, cache_values, cache_pos, mask,
                    recurrent_states=recurrent_states, conv_states=conv_states,
                    embed_w=embed_w, lm_head_w=lm_head_w,
                    final_norm_w=final_norm_w, layer_params=layer_params,
                    config=config, output_attentions=output_attentions,
                    per_head_attn=per_head_attn,
                )
        else:
            # Standard model: donate KV cache (args 1,2)
            @partial(jax.jit, donate_argnums=(1, 2))
            def step_fn(token_ids, cache_keys, cache_values, cache_pos, mask,
                        embed_w, lm_head_w, final_norm_w, layer_params):
                return _forward_jittable(
                    token_ids, cache_keys, cache_values, cache_pos, mask,
                    embed_w=embed_w, lm_head_w=lm_head_w,
                    final_norm_w=final_norm_w, layer_params=layer_params,
                    config=config, output_attentions=output_attentions,
                    per_head_attn=per_head_attn,
                )

        return step_fn

    def _get_decode_fn(self, with_attn: bool = False, per_head_attn: bool = False):
        """Get or create JIT-compiled single-token decode function."""
        if per_head_attn:
            if self._jit_decode_per_head is None:
                self._jit_decode_per_head = self._make_jit_fn(
                    output_attentions=True, per_head_attn=True)
            return self._jit_decode_per_head
        elif with_attn:
            if self._jit_decode_with_attn is None:
                self._jit_decode_with_attn = self._make_jit_fn(output_attentions=True)
            return self._jit_decode_with_attn
        else:
            if self._jit_decode is None:
                self._jit_decode = self._make_jit_fn(output_attentions=False)
            return self._jit_decode

    def _get_prefill_fn(self, bucket_size: int):
        """Get or create JIT-compiled prefill function for a specific bucket size.

        Prefill always returns attention scores (generators need them).
        Each bucket_size produces a separate compilation.
        """
        if bucket_size not in self._jit_prefill:
            self._jit_prefill[bucket_size] = self._make_jit_fn(output_attentions=True)
        return self._jit_prefill[bucket_size]

    def _init_cache(self) -> tuple[list[Any], list[Any], list[Any] | None, list[Any] | None]:
        """Create empty static-shape KV cache and DeltaNet state arrays.

        Returns (cache_keys, cache_values, recurrent_states, conv_states).
        recurrent_states and conv_states are None for non-hybrid models.
        When tensor-parallel mesh is active, shard cache along KV heads (axis 1).
        """
        n_kv = self.config.num_kv_layers
        shape = (1, self.config.num_key_value_heads, self.max_cache_len, self.config.head_dim)
        if self.mesh is not None:
            sharding = NamedSharding(self.mesh, P(None, "tp", None, None))
            keys = [jax.device_put(jnp.zeros(shape, dtype=jnp.bfloat16), sharding)
                    for _ in range(n_kv)]
            values = [jax.device_put(jnp.zeros(shape, dtype=jnp.bfloat16), sharding)
                      for _ in range(n_kv)]
        else:
            keys = [jnp.zeros(shape, dtype=jnp.bfloat16) for _ in range(n_kv)]
            values = [jnp.zeros(shape, dtype=jnp.bfloat16) for _ in range(n_kv)]

        recurrent_states = None
        conv_states = None
        if self.config.layer_types is not None:
            n_dn = self.config.num_linear_layers
            c = self.config
            rs_shape = (1, c.linear_num_v_heads, c.linear_head_k_dim, c.linear_head_v_dim)
            # conv1d operates on full QKV: key_dim*2 + value_dim channels
            qkv_dim = c.linear_num_k_heads * c.linear_head_k_dim * 2 + c.linear_num_v_heads * c.linear_head_v_dim
            cs_shape = (1, qkv_dim, c.linear_conv_kernel_dim)
            recurrent_states = [jnp.zeros(rs_shape, dtype=jnp.bfloat16) for _ in range(n_dn)]
            conv_states = [jnp.zeros(cs_shape, dtype=jnp.bfloat16) for _ in range(n_dn)]

        return keys, values, recurrent_states, conv_states

    def _build_eviction_mask(
        self, attention_mask: Sequence[float], cache_pos_arr: Any, seq_len: int
    ) -> Any:
        """Build combined eviction + position mask."""
        flags = list(attention_mask)
        if len(flags) > self.max_cache_len:
            raise ValueError(
                f"attention_mask length {len(flags)} exceeds max_cache_len {self.max_cache_len}"
            )
        if len(flags) < self.max_cache_len:
            flags = flags + [0.0] * (self.max_cache_len - len(flags))
        flags_arr = jnp.array(flags[:self.max_cache_len], dtype=jnp.float32)
        eviction_mask = jnp.where(flags_arr > 0.5, 0.0, -1e9)
        eviction_mask = eviction_mask[None, None, None, :]  # (1,1,1,max_cache)
        pos_mask = _build_position_mask(cache_pos_arr, seq_len, self.max_cache_len)
        return jnp.where(
            (pos_mask > -1e8) & (eviction_mask > -1e8),
            0.0, -1e9
        )

    def _build_per_head_eviction_mask(
        self, per_head_mask: np.ndarray, cache_pos_arr: Any, seq_len: int
    ) -> Any:
        """Build per-head eviction mask from (n_kv_heads, max_cache) float array.

        For GQA models, faithful policies naturally emit one mask per KV head, but
        the decode path flattens grouped attention logits to query-head shape before
        adding the mask. Expand KV-head masks across query-head groups so the mask
        matches the attention tensor shape.

        Returns (1, n_attention_heads, 1, max_cache) additive mask for per-head
        masking in GQA and (1, n_kv_heads, 1, max_cache) for full MHA.
        """
        n_kv_heads = per_head_mask.shape[0]
        n_attention_heads = self.config.num_attention_heads
        # Pad or truncate to max_cache_len
        if per_head_mask.shape[1] < self.max_cache_len:
            pad_width = self.max_cache_len - per_head_mask.shape[1]
            per_head_mask = np.pad(per_head_mask, ((0, 0), (0, pad_width)),
                                   constant_values=0.0)
        per_head_mask = per_head_mask[:, :self.max_cache_len]

        if n_attention_heads != n_kv_heads:
            if n_attention_heads % n_kv_heads != 0:
                raise ValueError(
                    "Per-head eviction mask shape is incompatible with grouped-query attention: "
                    f"{n_attention_heads} attention heads vs {n_kv_heads} KV heads."
                )
            n_groups = n_attention_heads // n_kv_heads
            per_head_mask = np.repeat(per_head_mask, n_groups, axis=0)

        flags_arr = jnp.array(per_head_mask, dtype=jnp.float32)  # (H, C)
        eviction_mask = jnp.where(flags_arr > 0.5, 0.0, -1e9)
        eviction_mask = eviction_mask[None, :, None, :]  # (1, H, 1, C)
        pos_mask = _build_position_mask(cache_pos_arr, seq_len, self.max_cache_len)
        # pos_mask is (1,1,1,C) for decode — broadcast across heads
        return jnp.where(
            (pos_mask > -1e8) & (eviction_mask > -1e8),
            0.0, -1e9
        )

    def __call__(
        self,
        *,
        input_ids: list[int] | Sequence[int],
        past_key_values: tuple | None = None,
        attention_mask: list[float] | Sequence[float] | None = None,
        output_attentions: bool = False,
        use_cache: bool = True,
    ) -> _ModelOutput:
        """Forward pass matching KVCoupledQwen35Generator interface.

        past_key_values is a tuple of:
          (cache_keys, cache_values, cache_pos) for standard models, or
          (cache_keys, cache_values, cache_pos, recurrent_states, conv_states)
          for hybrid models.
        """
        seq_len = len(input_ids)
        is_hybrid = self.config.layer_types is not None

        # Unpack or initialize cache
        if past_key_values is not None:
            cache_keys, cache_values, cache_pos = past_key_values[:3]
            recurrent_states = past_key_values[3] if is_hybrid and len(past_key_values) > 3 else None
            conv_states = past_key_values[4] if is_hybrid and len(past_key_values) > 4 else None
        else:
            cache_keys, cache_values, recurrent_states, conv_states = self._init_cache()
            cache_pos = 0

        cache_pos_arr = jnp.int32(cache_pos)

        # ── Decode path (seq_len == 1) ────────────────────────────────────
        if seq_len == 1:
            if cache_pos >= self.max_cache_len:
                raise ValueError(
                    f"cache_pos {cache_pos} >= max_cache_len {self.max_cache_len}. "
                    "Increase max_cache_len or reduce generation length."
                )
            ids_array = jnp.array([list(input_ids)], dtype=jnp.int32)

            # Detect per-head mode: attention_mask is a 2D numpy array (n_kv_heads, n_pos)
            use_per_head = (isinstance(attention_mask, np.ndarray)
                           and attention_mask.ndim == 2)

            if attention_mask is not None:
                if use_per_head:
                    mask = self._build_per_head_eviction_mask(
                        attention_mask, cache_pos_arr, 1)
                else:
                    mask = self._build_eviction_mask(attention_mask, cache_pos_arr, 1)
            else:
                mask = _build_position_mask(cache_pos_arr, 1, self.max_cache_len)

            if self.eager:
                result = _forward_jittable(
                    ids_array, cache_keys, cache_values, cache_pos_arr, mask,
                    recurrent_states=recurrent_states, conv_states=conv_states,
                    embed_w=self._embed_w, lm_head_w=self._lm_head_w,
                    final_norm_w=self._final_norm_w, layer_params=self._layer_params,
                    config=self.config, output_attentions=output_attentions,
                    per_head_attn=use_per_head,
                )
            else:
                if use_per_head:
                    fn = self._get_decode_fn(per_head_attn=True)
                else:
                    fn = self._get_decode_fn(with_attn=output_attentions)
                if is_hybrid:
                    result = fn(ids_array, cache_keys, cache_values, cache_pos_arr,
                                mask, recurrent_states, conv_states,
                                self._embed_w, self._lm_head_w,
                                self._final_norm_w, self._layer_params)
                else:
                    result = fn(ids_array, cache_keys, cache_values, cache_pos_arr, mask,
                                self._embed_w, self._lm_head_w,
                                self._final_norm_w, self._layer_params)
            logits, new_keys, new_vals = result[0], result[1], result[2]
            stacked_attn = result[3] if output_attentions else None
            new_rs = result[4] if is_hybrid else None
            new_cs = result[5] if is_hybrid else None

            new_cache_pos = cache_pos + 1
            if use_cache:
                if is_hybrid:
                    new_past = (new_keys, new_vals, new_cache_pos, new_rs, new_cs)
                else:
                    new_past = (new_keys, new_vals, new_cache_pos)
            else:
                new_past = None

            logits_np = np.asarray(logits)
            attn_np = None
            if output_attentions and stacked_attn is not None:
                if use_per_head:
                    # stacked_attn: (L_kv, B, H, S=1, C) — per-head scores preserved
                    # Reduce across layers (max), keep heads: (H, C)
                    per_head_jax = jnp.max(
                        stacked_attn[:, 0, :, 0, :], axis=0)  # (H, C)
                    per_head_np = np.asarray(per_head_jax)
                    attn_np = _AttentionTuple([per_head_np])
                else:
                    # stacked_attn: (L_kv, B, 1, C) — heads already reduced inside JIT.
                    # Collapse to per-key max on device, then single D2H transfer.
                    per_key_jax = jnp.max(
                        stacked_attn[:, 0, 0, :], axis=0)  # (max_cache_len,)
                    per_key_np = np.asarray(per_key_jax)  # single D2H transfer
                    attn_np = _AttentionTuple([per_key_np.reshape(1, 1, 1, -1)])

            return _ModelOutput(logits=logits_np, past_key_values=new_past, attentions=attn_np)

        # ── Prefill path (seq_len > 1) — chunked to fit single-chip HBM ──
        if seq_len == 0:
            raise ValueError("input_ids must be non-empty")
        chunk = self.prefill_chunk_size
        ids_list = list(input_ids)

        # Accumulator for per-layer max attention scores (from KV layers only)
        n_kv_layers = self.config.num_kv_layers
        attn_accum: list[np.ndarray] | None = None
        if output_attentions:
            attn_accum = [np.full(self.max_cache_len, -np.inf) for _ in range(n_kv_layers)]

        cur_pos = cache_pos
        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            chunk_ids = ids_list[start:end]
            real_len = end - start

            effective_len = real_len
            if not self.eager and real_len < chunk:
                chunk_ids = chunk_ids + [0] * (chunk - real_len)
                effective_len = chunk

            # Guard: ensure padded chunk won't exceed cache capacity
            # (JAX dynamic_update_slice clamps start index, which would
            #  silently overwrite earlier cache positions)
            if cur_pos + effective_len > self.max_cache_len:
                raise ValueError(
                    f"Prefill chunk would exceed cache: "
                    f"cur_pos={cur_pos} + effective_len={effective_len} > "
                    f"max_cache_len={self.max_cache_len}. Truncate input."
                )

            ids_array = jnp.array([chunk_ids], dtype=jnp.int32)
            cur_pos_arr = jnp.int32(cur_pos)

            if attention_mask is not None:
                mask = self._build_eviction_mask(attention_mask, cur_pos_arr, effective_len)
            else:
                mask = _build_position_mask(cur_pos_arr, effective_len, self.max_cache_len)

            if self.eager:
                result = _forward_jittable(
                    ids_array, cache_keys, cache_values, cur_pos_arr, mask,
                    recurrent_states=recurrent_states, conv_states=conv_states,
                    embed_w=self._embed_w, lm_head_w=self._lm_head_w,
                    final_norm_w=self._final_norm_w, layer_params=self._layer_params,
                    config=self.config, output_attentions=output_attentions,
                )
                logits = result[0]
                cache_keys, cache_values = result[1], result[2]
                chunk_attn = result[3] if output_attentions else None
                if is_hybrid:
                    recurrent_states, conv_states = result[4], result[5]
            else:
                fn = self._get_prefill_fn(chunk)
                if is_hybrid:
                    result = fn(ids_array, cache_keys, cache_values, cur_pos_arr,
                                mask, recurrent_states, conv_states,
                                self._embed_w, self._lm_head_w,
                                self._final_norm_w, self._layer_params)
                else:
                    result = fn(ids_array, cache_keys, cache_values, cur_pos_arr, mask,
                                self._embed_w, self._lm_head_w,
                                self._final_norm_w, self._layer_params)
                logits, cache_keys, cache_values = result[0], result[1], result[2]
                chunk_attn = result[3] if output_attentions else None
                if is_hybrid:
                    recurrent_states, conv_states = result[4], result[5]

            # Merge attention scores: keep running per-key max across chunks.
            if output_attentions and chunk_attn is not None and attn_accum is not None:
                sliced = chunk_attn[:, :, :real_len, :]
                per_layer_max = jnp.max(sliced, axis=(1, 2))
                per_layer_max_np = np.asarray(per_layer_max)
                for li in range(n_kv_layers):
                    attn_accum[li] = np.maximum(attn_accum[li], per_layer_max_np[li])

            cur_pos += real_len
            if cur_pos > self.max_cache_len:
                raise ValueError(
                    f"Prefill would write past cache: cur_pos={cur_pos} > "
                    f"max_cache_len={self.max_cache_len}. Truncate input."
                )
            last_real_len = real_len

        new_cache_pos = cur_pos
        if use_cache:
            if is_hybrid:
                new_past = (cache_keys, cache_values, new_cache_pos, recurrent_states, conv_states)
            else:
                new_past = (cache_keys, cache_values, new_cache_pos)
        else:
            new_past = None

        if output_attentions and attn_accum is not None:
            for li in range(n_kv_layers):
                attn_accum[li][new_cache_pos:] = 0.0

        logits_np = np.asarray(logits[:, :last_real_len, :])

        attn_np = None
        if output_attentions and attn_accum is not None:
            fake_layers = [a.reshape(1, 1, 1, -1) for a in attn_accum]
            attn_np = _AttentionTuple(fake_layers)

        return _ModelOutput(logits=logits_np, past_key_values=new_past, attentions=attn_np)


# ─── Tokenizer adapter ───────────────────────────────────────────────────────

class JAXTokenizerAdapter:
    """Wraps a HuggingFace tokenizer for the KVCoupledQwen35Generator interface."""

    def __init__(self, tokenizer: Any):
        self._tok = tokenizer

    def decode(self, ids: list[int]) -> str:
        return str(self._tok.decode(ids, skip_special_tokens=True))


# ─── Convenience loader ──────────────────────────────────────────────────────

def load_jax_model_and_tokenizer(
    model_id_or_path: str,
    max_cache_len: int = 4096,
    eager: bool = False,
    prefill_chunk_size: int | None = None,
    tp_size: int | None = None,
) -> tuple[JAXModelAdapter, JAXTokenizerAdapter]:
    """Load a model + tokenizer ready for KVCoupledQwen35Generator.

    Args:
        model_id_or_path: HuggingFace model ID or local path.
            For HF cached models, resolves the snapshot directory automatically.
        max_cache_len: maximum positions in KV cache.
        eager: if True, skip JIT compilation (for models that OOM during JIT).
        prefill_chunk_size: tokens per prefill chunk (default: 256; use 32 for 7B models).
        tp_size: number of devices for tensor parallelism (None=single device).

    Returns:
        (model_adapter, tokenizer_adapter) tuple.
    """
    from transformers import AutoTokenizer

    model_path = Path(model_id_or_path)

    # If given a HF model ID (not a path), resolve from HF cache
    if not model_path.exists():
        model_path = _resolve_hf_cache(model_id_or_path)

    print(f"[JAX] Loading config from {model_path} ...")
    config = TransformerConfig.from_hf_config(model_path / "config.json")
    print(f"[JAX] Architecture: {config.arch}, "
          f"{config.num_hidden_layers} layers, "
          f"{config.hidden_size}d, "
          f"{config.num_attention_heads}Q/{config.num_key_value_heads}KV heads")
    if config.layer_types is not None:
        print(f"[JAX] Hybrid: {config.num_kv_layers} KV layers, "
              f"{config.num_linear_layers} DeltaNet layers, "
              f"partial_rotary={config.partial_rotary_factor}")

    # Set up tensor parallelism mesh if requested.
    # TP size must divide both num_attention_heads and num_key_value_heads.
    # Auto-clamp to the largest valid TP ≤ requested.
    effective_tp = tp_size
    if tp_size is not None and tp_size > 1:
        max_valid_tp = tp_size
        while max_valid_tp > 1:
            if (config.num_attention_heads % max_valid_tp == 0 and
                    config.num_key_value_heads % max_valid_tp == 0 and
                    config.intermediate_size % max_valid_tp == 0):
                break
            max_valid_tp -= 1
        if max_valid_tp != tp_size:
            print(f"[JAX] TP={tp_size} incompatible with {config.num_key_value_heads} KV heads / "
                  f"{config.num_attention_heads} Q heads. Clamped to TP={max_valid_tp}.")
        effective_tp = max_valid_tp

    mesh = create_tp_mesh(effective_tp)
    if mesh is not None:
        print(f"[JAX] Tensor parallelism: {effective_tp} devices, mesh={mesh}")

    print(f"[JAX] Loading weights from safetensors ...")
    raw_params = _load_all_safetensors(model_path)

    # Normalize VLM-style key prefixes (e.g. Qwen3.5):
    #   model.language_model.layers.X.* → model.layers.X.*
    #   Also strip model.visual.* (vision encoder) and mtp.* params
    has_lm_prefix = any(k.startswith("model.language_model.") for k in raw_params)
    if has_lm_prefix:
        normalized: dict[str, Any] = {}
        n_vision = 0
        for k, v in raw_params.items():
            if k.startswith("model.language_model."):
                new_key = "model." + k[len("model.language_model."):]
                normalized[new_key] = v
            elif k.startswith("model.visual."):
                n_vision += 1
            elif k.startswith("lm_head."):
                normalized[k] = v
            # Skip mtp.* and other non-text params
        if n_vision > 0:
            print(f"[JAX] Skipped {n_vision} vision encoder params")
        del raw_params
        raw_params = normalized

    # Normalize Gemma3-style key prefixes:
    #   language_model.model.* → model.*
    has_gemma3_prefix = any(k.startswith("language_model.model.") for k in raw_params)
    if has_gemma3_prefix:
        normalized = {}
        n_skipped_g3 = 0
        for k, v in raw_params.items():
            if k.startswith("language_model.model."):
                new_key = "model." + k[len("language_model.model."):]
                normalized[new_key] = v
            elif k.startswith("language_model.lm_head."):
                new_key = k[len("language_model."):]  # → "lm_head.weight"
                normalized[new_key] = v
            elif k.startswith(("vision_tower.", "multi_modal_projector.")):
                n_skipped_g3 += 1
            elif k.startswith("lm_head."):
                normalized[k] = v
            else:
                n_skipped_g3 += 1
        if n_skipped_g3 > 0:
            print(f"[JAX] Skipped {n_skipped_g3} non-text params (vision/projector)")
        del raw_params
        raw_params = normalized

    # Filter to text-model weights only (skip vision encoder for VLMs)
    text_prefixes = ("model.", "lm_head.")
    text_params = {k: v for k, v in raw_params.items()
                   if any(k.startswith(p) for p in text_prefixes)}
    n_skipped = len(raw_params) - len(text_params)
    if n_skipped > 0:
        print(f"[JAX] Skipped {n_skipped} non-text params")
    del raw_params

    # Convert to JAX arrays. When TP is active, shard directly from numpy
    # to avoid placing 27B+ model entirely on a single device (OOM).
    if mesh is not None:
        disable_mh_assert = False
        if os.environ.get("CFKVE_DISABLE_MH_ASSERT_EQUAL", "0") == "1":
            if jax.process_count() > 1:
                local_proc = int(jax.process_index())
                mesh_devs = list(mesh.devices.flat)
                disable_mh_assert = any(int(d.process_index) != local_proc for d in mesh_devs)

        mh_mod = None
        mh_assert_orig = None
        if disable_mh_assert:
            # JAX device_put on multi-host sharding validates host inputs with
            # process allgather. For large params this can OOM before sharding.
            # In this code path each host reads identical checkpoint shards.
            from jax.experimental import multihost_utils as _mh  # local import to limit scope
            mh_mod = _mh
            mh_assert_orig = _mh.assert_equal
            _mh.assert_equal = lambda *args, **kwargs: None

        print(f"[JAX] Sharding {len(text_params)} params across {effective_tp} devices ...")
        params = {}
        try:
            for k, v in text_params.items():
                sharded = _shard_weight(
                    jnp.asarray(v, dtype=jnp.bfloat16), k, config, mesh)
                params[k] = sharded
                del v, sharded  # free numpy + intermediate immediately
        finally:
            if mh_mod is not None and mh_assert_orig is not None:
                mh_mod.assert_equal = mh_assert_orig
        del text_params
        print(f"[JAX] Sharding complete")
    else:
        print(f"[JAX] Converting {len(text_params)} tensors to bf16 JAX arrays ...")
        params = {k: jnp.asarray(v, dtype=jnp.bfloat16) for k, v in text_params.items()}
        del text_params

    print(f"[JAX] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    model_adapter = JAXModelAdapter(
        config, params, max_cache_len=max_cache_len, eager=eager,
        prefill_chunk_size=prefill_chunk_size, mesh=mesh,
    )
    tok_adapter = JAXTokenizerAdapter(tokenizer)
    tp_str = f", TP={effective_tp}" if mesh else ""
    mode_str = "eager" if eager else "JIT"
    print(f"[JAX] Ready ({mode_str}{tp_str}). Devices: {[str(d) for d in jax.local_devices()[:effective_tp or 1]]}")
    return model_adapter, tok_adapter


def _resolve_hf_cache(model_id: str) -> Path:
    """Resolve a HuggingFace model ID to its cached snapshot directory."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_slug = f"models--{model_id.replace('/', '--')}"
    model_dir = cache_dir / model_slug

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model '{model_id}' not found in HF cache at {model_dir}. "
            f"Download it first with: huggingface-cli download {model_id}"
        )

    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"No snapshots directory in {model_dir}")

    # Use the most recent snapshot
    snapshot_dirs = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshots in {snapshots}")

    return snapshot_dirs[0]
