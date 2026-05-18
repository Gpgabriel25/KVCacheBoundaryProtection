from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
from pathlib import Path
import struct
from typing import Any, Callable, Mapping

import numpy as np


class MissingDependencyError(RuntimeError):
    """Raised when an optional runtime dependency required by this module is missing."""


@dataclasses.dataclass(frozen=True)
class SafetensorsTensorMetadata:
    """Lightweight tensor metadata extracted from a safetensors header."""

    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclasses.dataclass(frozen=True)
class TensorSpecExpectation:
    """Expected metadata for a tensor, used by compatibility checks."""

    shape: tuple[int, ...] | None = None
    dtype: str | None = None


@dataclasses.dataclass(frozen=True)
class TensorSpecMismatch:
    """A single tensor field mismatch discovered during compatibility checks."""

    name: str
    field: str
    expected: Any
    actual: Any


@dataclasses.dataclass(frozen=True)
class CheckpointCompatibilityResult:
    """Structured result of expected-tensor compatibility checks."""

    is_compatible: bool
    missing_required: tuple[str, ...]
    missing_expected_specs: tuple[str, ...]
    mismatches: tuple[TensorSpecMismatch, ...]


def _import_optional_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            f"Missing optional dependency '{module_name}'. Install it with: {install_hint}"
        ) from exc


def _parse_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Invalid safetensors file (missing header length): {path}")

        header_length = int(struct.unpack("<Q", prefix)[0])
        header_payload = handle.read(header_length)
        if len(header_payload) != header_length:
            raise ValueError(f"Invalid safetensors file (truncated header): {path}")

    try:
        parsed = json.loads(header_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid safetensors file (bad JSON header): {path}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid safetensors file (header is not an object): {path}")

    return parsed


def inspect_safetensors_metadata(
    path: str | Path,
    *,
    tensor_prefix: str | None = None,
) -> dict[str, SafetensorsTensorMetadata]:
    """Inspect tensor names, shapes, and dtypes from a safetensors header.

    This helper does not load tensor payloads into host memory and can be used
    before deciding whether full JAX placement should happen.
    """
    safetensors_path = Path(path)
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"Safetensors file not found: {safetensors_path}")

    header = _parse_safetensors_header(safetensors_path)
    metadata: dict[str, SafetensorsTensorMetadata] = {}

    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if tensor_prefix and not name.startswith(tensor_prefix):
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid safetensors header entry for tensor '{name}'.")

        dtype_raw = entry.get("dtype")
        shape_raw = entry.get("shape")
        if not isinstance(dtype_raw, str):
            raise ValueError(f"Invalid safetensors header dtype for tensor '{name}'.")
        if not isinstance(shape_raw, list):
            raise ValueError(f"Invalid safetensors header shape for tensor '{name}'.")

        metadata[name] = SafetensorsTensorMetadata(
            name=name,
            shape=tuple(int(dim) for dim in shape_raw),
            dtype=dtype_raw,
        )

    return metadata


def _coerce_tensor_spec_expectation(
    value: TensorSpecExpectation | Mapping[str, Any],
) -> TensorSpecExpectation:
    if isinstance(value, TensorSpecExpectation):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "Expected tensor spec values must be TensorSpecExpectation or mappings. "
            f"Received {type(value).__name__}."
        )

    shape_raw = value.get("shape")
    dtype_raw = value.get("dtype")
    shape: tuple[int, ...] | None
    if shape_raw is None:
        shape = None
    else:
        if not isinstance(shape_raw, (tuple, list)):
            raise TypeError("Expected shape must be a tuple or list when provided.")
        shape = tuple(int(dim) for dim in shape_raw)

    if dtype_raw is not None and not isinstance(dtype_raw, str):
        raise TypeError("Expected dtype must be a string when provided.")

    return TensorSpecExpectation(shape=shape, dtype=dtype_raw)


def check_safetensors_compatibility(
    metadata: Mapping[str, SafetensorsTensorMetadata],
    *,
    required_names: set[str] | None = None,
    expected_specs: Mapping[str, TensorSpecExpectation | Mapping[str, Any]] | None = None,
    check_shapes: bool = True,
    check_dtypes: bool = False,
) -> CheckpointCompatibilityResult:
    """Check checkpoint metadata against required names and optional tensor specs."""
    required = set(required_names or set())
    available = set(metadata.keys())
    missing_required = tuple(sorted(required - available))

    spec_entries = expected_specs or {}
    missing_expected_specs: list[str] = []
    mismatches: list[TensorSpecMismatch] = []

    for name in sorted(spec_entries.keys()):
        expected = _coerce_tensor_spec_expectation(spec_entries[name])
        if name not in metadata:
            missing_expected_specs.append(name)
            continue

        actual = metadata[name]
        if check_shapes and expected.shape is not None and tuple(actual.shape) != tuple(expected.shape):
            mismatches.append(
                TensorSpecMismatch(
                    name=name,
                    field="shape",
                    expected=tuple(expected.shape),
                    actual=tuple(actual.shape),
                )
            )
        if check_dtypes and expected.dtype is not None and str(actual.dtype) != expected.dtype:
            mismatches.append(
                TensorSpecMismatch(
                    name=name,
                    field="dtype",
                    expected=expected.dtype,
                    actual=str(actual.dtype),
                )
            )

    is_compatible = not missing_required and not missing_expected_specs and not mismatches
    return CheckpointCompatibilityResult(
        is_compatible=is_compatible,
        missing_required=missing_required,
        missing_expected_specs=tuple(missing_expected_specs),
        mismatches=tuple(mismatches),
    )


def check_safetensors_file_compatibility(
    path: str | Path,
    *,
    tensor_prefix: str | None = None,
    required_names: set[str] | None = None,
    expected_specs: Mapping[str, TensorSpecExpectation | Mapping[str, Any]] | None = None,
    check_shapes: bool = True,
    check_dtypes: bool = False,
) -> CheckpointCompatibilityResult:
    """Inspect a safetensors file and run metadata compatibility checks."""
    metadata = inspect_safetensors_metadata(path, tensor_prefix=tensor_prefix)
    return check_safetensors_compatibility(
        metadata,
        required_names=required_names,
        expected_specs=expected_specs,
        check_shapes=check_shapes,
        check_dtypes=check_dtypes,
    )


def filter_tensors_by_prefix(tensors: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Return tensors with keys that start with ``prefix``."""
    return {name: value for name, value in tensors.items() if name.startswith(prefix)}


def map_tensors(
    tensors: Mapping[str, Any],
    fn: Callable[[str, Any], Any],
) -> dict[str, Any]:
    """Apply ``fn`` to each tensor while preserving tensor names."""
    return {name: fn(name, value) for name, value in tensors.items()}


LayoutResolver = Callable[[str, tuple[int, ...]], Any]

PLACEMENT_MODE_HOST_LOCAL = "host_local"
PLACEMENT_MODE_NAMED_SHARDING = "named_sharding"
_VALID_PLACEMENT_MODES = {
    PLACEMENT_MODE_HOST_LOCAL,
    PLACEMENT_MODE_NAMED_SHARDING,
}


def qwen_partition_spec_resolver(name: str, _shape: tuple[int, ...]) -> tuple[Any, ...] | None:
    """Return practical PartitionSpec tuples for common Qwen tensor names.

    The returned value is intentionally tuple-based so it remains compatible with
    ``_coerce_partition_spec`` and does not require JAX at definition time.
    """
    normalized = name.lower()

    if normalized.endswith(("embed_tokens.weight", "tok_embeddings.weight")):
        return (None, "model")

    if normalized.endswith(("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight")):
        return ("model", None)

    if normalized.endswith("self_attn.o_proj.weight"):
        return (None, "model")

    if normalized.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
        return ("model", None)

    if normalized.endswith("mlp.down_proj.weight"):
        return (None, "model")

    if normalized.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", "norm.weight")):
        return None

    if normalized.endswith("lm_head.weight"):
        return ("model", None)

    return None


def create_single_process_mesh(jax_module: Any, *, axis_name: str = "data") -> Any:
    """Build a 1D mesh from all visible local devices for single-process runs."""
    if not hasattr(jax_module, "sharding") or not hasattr(jax_module.sharding, "Mesh"):
        raise MissingDependencyError(
            "JAX sharding APIs are unavailable. Use placement_mode='host_local' or upgrade JAX."
        )

    devices = list(jax_module.devices())
    if not devices:
        raise ValueError("No JAX devices available to create a mesh.")

    device_array = np.asarray(devices).reshape((len(devices),))
    return jax_module.sharding.Mesh(device_array, (axis_name,))


def _coerce_partition_spec(jax_module: Any, spec: Any) -> Any:
    partition_spec_type = jax_module.sharding.PartitionSpec
    if spec is None:
        return partition_spec_type()
    if isinstance(spec, partition_spec_type):
        return spec
    if isinstance(spec, (tuple, list)):
        return partition_spec_type(*spec)
    raise ValueError(
        "Layout resolver must return None, PartitionSpec, tuple, or list. "
        f"Received {type(spec).__name__}."
    )


def _validate_placement_mode(placement_mode: str) -> None:
    if placement_mode not in _VALID_PLACEMENT_MODES:
        valid_modes = ", ".join(sorted(_VALID_PLACEMENT_MODES))
        raise ValueError(
            f"Invalid placement_mode '{placement_mode}'. Expected one of: {valid_modes}."
        )


def _place_tensor(
    name: str,
    value: Any,
    *,
    jax_module: Any,
    placement_mode: str,
    layout_resolver: LayoutResolver | None,
    mesh: Any | None,
    mesh_axis_name: str,
) -> Any:
    host_value = jax_module.numpy.asarray(value)

    if placement_mode == PLACEMENT_MODE_HOST_LOCAL:
        return host_value

    if not hasattr(jax_module, "sharding"):
        raise MissingDependencyError(
            "JAX sharding APIs are unavailable. Use placement_mode='host_local' or upgrade JAX."
        )

    for attr in ("NamedSharding", "PartitionSpec", "Mesh"):
        if not hasattr(jax_module.sharding, attr):
            raise MissingDependencyError(
                "JAX sharding APIs are unavailable. "
                "Use placement_mode='host_local' or upgrade JAX."
            )

    active_mesh = mesh if mesh is not None else create_single_process_mesh(
        jax_module,
        axis_name=mesh_axis_name,
    )

    spec = None
    if layout_resolver is not None:
        shape = tuple(int(dim) for dim in host_value.shape)
        spec = layout_resolver(name, shape)
    partition_spec = _coerce_partition_spec(jax_module, spec)
    sharding = jax_module.sharding.NamedSharding(active_mesh, partition_spec)
    return jax_module.device_put(host_value, sharding)


def load_safetensors_as_jax(
    path: str | Path,
    *,
    tensor_prefix: str | None = None,
    placement_mode: str = PLACEMENT_MODE_HOST_LOCAL,
    layout_resolver: LayoutResolver | None = None,
    mesh: Any | None = None,
    mesh_axis_name: str = "data",
) -> dict[str, Any]:
    """Load tensors from a safetensors file and convert them to JAX arrays."""
    _validate_placement_mode(placement_mode)

    safetensors_path = Path(path)
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"Safetensors file not found: {safetensors_path}")

    safetensors_numpy = _import_optional_dependency(
        "safetensors.numpy",
        "pip install safetensors",
    )
    jax = _import_optional_dependency(
        "jax",
        "pip install jax",
    )

    tensors = safetensors_numpy.load_file(str(safetensors_path))
    if tensor_prefix:
        tensors = filter_tensors_by_prefix(tensors, tensor_prefix)

    return map_tensors(
        tensors,
        lambda name, value: _place_tensor(
            name,
            value,
            jax_module=jax,
            placement_mode=placement_mode,
            layout_resolver=layout_resolver,
            mesh=mesh,
            mesh_axis_name=mesh_axis_name,
        ),
    )


def load_safetensors_subset_as_jax(
    path: str | Path,
    *,
    tensor_prefix: str | None = None,
    max_tensors: int = 8,
    placement_mode: str = PLACEMENT_MODE_HOST_LOCAL,
    layout_resolver: LayoutResolver | None = None,
    mesh: Any | None = None,
    mesh_axis_name: str = "data",
) -> dict[str, Any]:
    """Load at most ``max_tensors`` tensors and convert them to JAX arrays.

    This bounded helper is intended for lightweight integration smoke checks
    where loading the full checkpoint would be unnecessarily expensive.
    """
    _validate_placement_mode(placement_mode)
    if max_tensors <= 0:
        raise ValueError("max_tensors must be > 0")

    safetensors_path = Path(path)
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"Safetensors file not found: {safetensors_path}")

    safetensors = _import_optional_dependency(
        "safetensors",
        "pip install safetensors",
    )
    jax = _import_optional_dependency(
        "jax",
        "pip install jax",
    )

    selected_names: list[str] = []
    with safetensors.safe_open(str(safetensors_path), framework="np") as handle:
        names = [name for name in handle.keys() if (not tensor_prefix or name.startswith(tensor_prefix))]
        selected_names = sorted(names)[:max_tensors]
        selected_values = {name: handle.get_tensor(name) for name in selected_names}

    return map_tensors(
        selected_values,
        lambda name, value: _place_tensor(
            name,
            value,
            jax_module=jax,
            placement_mode=placement_mode,
            layout_resolver=layout_resolver,
            mesh=mesh,
            mesh_axis_name=mesh_axis_name,
        ),
    )


def summarize_safetensors(
    path: str | Path,
    *,
    tensor_prefix: str | None = None,
    max_entries: int = 8,
) -> list[str]:
    """Load tensors and return a compact shape/dtype summary for smoke checks."""
    tensors = load_safetensors_as_jax(path, tensor_prefix=tensor_prefix)
    lines = [f"Loaded {len(tensors)} tensors from {Path(path)}"]

    for name in sorted(tensors.keys())[:max_entries]:
        value = tensors[name]
        shape = tuple(int(dim) for dim in value.shape)
        lines.append(f"{name}: shape={shape} dtype={value.dtype}")

    if len(tensors) > max_entries:
        lines.append(f"... {len(tensors) - max_entries} additional tensors omitted")

    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-check a safetensors file by loading it as JAX arrays and printing a summary."
    )
    parser.add_argument("path", type=Path, help="Path to a .safetensors file")
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional tensor name prefix filter.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=8,
        help="Maximum number of tensor summaries to print.",
    )
    args = parser.parse_args(argv)

    for line in summarize_safetensors(
        args.path,
        tensor_prefix=args.prefix,
        max_entries=args.max_entries,
    ):
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
