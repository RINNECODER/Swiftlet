"""Generate GLM linear-attention oracles from the pinned local mlx-vlm sources.

The numerical expected values come from executing `Glm5NextLinearAttention` and
`gated_delta_update` (ops path, `use_kernel=False`) extracted from
../glm-reference. This script does not download weights or modify that tree.

Run:
  ../glm-reference/.venv/bin/python scripts/gen_glm_linear_attention_fixtures.py
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

REVISION = "a74c7de90a344a2c2c7334acb4e48b57a40480e2"
ROOT = Path(__file__).resolve().parents[1]
REF = ROOT.parent / "glm-reference"
OUT = ROOT / "fixtures" / "glm-linear-attention" / "reference.json"


def series(count, salt, scale):
    return [
        (((i * 17 + salt * 13) % 97) / 97.0 - 0.5) * 2 * scale
        for i in range(count)
    ]


def flatten_f32(array):
    import mlx.core as mx

    value = array.astype(mx.float32).reshape(-1)
    mx.eval(value)
    return value.tolist()


def require_close(name, actual, expected, tolerance=1e-5):
    import mlx.core as mx

    mx.eval(actual, expected)
    if actual.shape != expected.shape:
        raise SystemExit(f"{name} shape {actual.shape} != {expected.shape}")
    if actual.size == 0 and expected.size == 0:
        return 0.0
    error = float(mx.max(mx.abs(actual - expected)))
    if error > tolerance:
        raise SystemExit(f"{name} diverged by {error}")
    return error


def assert_f32_roundtrip(values, original):
    import mlx.core as mx

    restored = mx.array(json.loads(json.dumps(values)), dtype=mx.float32).reshape(original.shape)
    mx.eval(original)
    if not mx.array_equal(restored, original.astype(mx.float32)):
        raise SystemExit("JSON encoding did not round-trip float32 values")


class StepCache:
    """Minimal cache satisfying the calls in ShortConv1d and gated_delta_update."""

    def __init__(self):
        self.slots = [None, None]
        self.lengths = None

    def __getitem__(self, index):
        return self.slots[index]

    def update_window(self, index, values, keep, lengths=None):
        # keep == 0 must stay empty. A Python slice of -0 is the whole axis.
        if keep <= 0:
            self.slots[index] = values[:, 0:0, :]
        else:
            self.slots[index] = values[:, -keep:, :]

    def update_recurrent(self, index, length, function):
        output = function(self.slots[index], None)
        self.slots[index] = output[1]
        return output[0], output[1]

    def advance(self, count):
        return None


def load_upstream():
    import mlx.core as mx
    import mlx.nn as nn
    from functools import partial
    from typing import Optional, Tuple

    language = (REF / "language.py").read_text()
    gated = (REF / "gated_delta.py").read_text()
    language_tree = ast.parse(language)
    gated_tree = ast.parse(gated)
    function_names = {
        "compute_g",
        "compute_g_safe",
        "gated_delta_ops",
        "gated_delta_update",
        "_gated_delta_step_ops",
    }
    functions = [
        node for node in gated_tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    if len(functions) != len(function_names):
        raise SystemExit("gated_delta.py is missing expected functions")
    gated_ns = {
        "mx": mx,
        "nn": nn,
        "Optional": Optional,
        "Tuple": Tuple,
        "partial": partial,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(REF / "gated_delta.py"), "exec"), gated_ns)

    def linear(layer, value):
        # Unquantized bias-free nn.Linear. The extracted module builds only those.
        projected = value @ layer.weight.T
        if "bias" in layer:
            projected = projected + layer.bias
        return projected

    def gated_delta_update(*args, **kwargs):
        kwargs["use_kernel"] = False
        return gated_ns["gated_delta_update"](*args, **kwargs)

    class_names = {"ShortConv1d", "Glm5NextLinearAttention"}
    classes = [
        node for node in language_tree.body
        if isinstance(node, ast.ClassDef) and node.name in class_names
    ]
    if len(classes) != len(class_names):
        raise SystemExit("language.py is missing Glm5NextLinearAttention or ShortConv1d")
    language_ns = {
        "mx": mx,
        "nn": nn,
        "linear": linear,
        "gated_delta_update": gated_delta_update,
        "TextConfig": object,
    }
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(REF / "language.py"), "exec"), language_ns)
    return {
        "mx": mx,
        "nn": nn,
        "Attention": language_ns["Glm5NextLinearAttention"],
        "compute_g_safe": gated_ns["compute_g_safe"],
        "linear": linear,
        "language_sha256": hashlib.sha256(language.encode()).hexdigest(),
        "gated_delta_sha256": hashlib.sha256(gated.encode()).hexdigest(),
    }


def assign(model, spec):
    import mlx.core as mx

    def put(shape, salt, scale, bias=0.0):
        count = 1
        for axis in shape:
            count *= axis
        return mx.array(series(count, salt, scale), dtype=mx.float32).reshape(shape) + bias

    model.qkv_proj.weight = put(tuple(model.qkv_proj.weight.shape), spec["salt"], spec["qkv_scale"])
    model.qkv_conv.conv.weight = put(tuple(model.qkv_conv.conv.weight.shape), spec["salt"] + 1, spec["conv_scale"])
    model.fbg_a_proj.weight = put(tuple(model.fbg_a_proj.weight.shape), spec["salt"] + 2, spec["fbg_scale"])
    model.f_b_proj.weight = put(tuple(model.f_b_proj.weight.shape), spec["salt"] + 3, spec["fb_scale"])
    model.g_b_proj.weight = put(tuple(model.g_b_proj.weight.shape), spec["salt"] + 4, spec["gb_scale"])
    model.A_log = mx.array(spec["a_log"], dtype=mx.float32)
    model.dt_bias = mx.array(spec["dt_bias"], dtype=mx.float32)
    model.o_norm.weight = put((spec["head_dim"],), spec["salt"] + 5, spec["norm_scale"], bias=1.0)
    model.o_proj.weight = put(tuple(model.o_proj.weight.shape), spec["salt"] + 6, spec["out_scale"])


def manual_short_conv(weight, rows):
    """Depthwise causal conv + silu, mirroring ShortConv1d's zero history."""
    import mlx.core as mx
    import mlx.nn as nn

    channels, kernel, _ = weight.shape
    taps = weight.reshape(channels, kernel)
    history = [mx.zeros((channels,), dtype=mx.float32) for _ in range(kernel - 1)]
    outputs = []
    states = []
    for row in rows:
        window = history + [row]
        accumulated = mx.zeros((channels,), dtype=mx.float32)
        for lag in range(kernel):
            accumulated = accumulated + taps[:, lag] * window[lag]
        outputs.append(nn.silu(accumulated))
        history = window[1:] if kernel > 1 else []
        if kernel == 1:
            states.append(mx.zeros((0, channels), dtype=mx.float32))
        else:
            states.append(mx.stack(history, axis=0))
    return outputs, states


def build_case(upstream, spec):
    import mlx.core as mx
    from types import SimpleNamespace

    config = SimpleNamespace(
        hidden_size=spec["hidden_size"],
        linear_num_heads=spec["num_heads"],
        linear_head_dim=spec["head_dim"],
        linear_conv_kernel_dim=spec["conv_kernel_size"],
        linear_lower_bound=spec["lower_bound"],
        rms_norm_eps=spec["rms_norm_eps"],
    )
    model = upstream["Attention"](config, 0)
    assign(model, spec)
    tokens = mx.stack([
        mx.array(series(spec["hidden_size"], spec["salt"] + 20 + index, spec["token_scale"]), dtype=mx.float32)
        for index in range(spec["tokens"])
    ])[None]
    full = model(tokens)
    cache = StepCache()
    stepped = []
    states = []
    for index in range(spec["tokens"]):
        stepped.append(model(tokens[:, index:index + 1], cache=cache))
        mx.eval(stepped[-1], cache.slots[0], cache.slots[1])
        states.append((cache.slots[0], cache.slots[1]))
    stepped_y = mx.concatenate(stepped, axis=1)
    require_close("full vs single-token", full, stepped_y)

    prefix = spec["prefix_length"]
    chunk_cache = StepCache()
    head = model(tokens[:, :prefix], cache=chunk_cache)
    require_close("chunk conv state", chunk_cache.slots[0], states[prefix - 1][0])
    require_close("chunk recurrent state", chunk_cache.slots[1], states[prefix - 1][1])
    tail = model(tokens[:, prefix:], cache=chunk_cache)
    require_close("chunk vs full", mx.concatenate([head, tail], axis=1), full)
    require_close("chunk final conv", chunk_cache.slots[0], states[-1][0])
    require_close("chunk final recurrent", chunk_cache.slots[1], states[-1][1])

    whole_cache = StepCache()
    whole = model(tokens, cache=whole_cache)
    require_close("whole-sequence cache output", whole, full)
    require_close("whole-sequence cache state", whole_cache.slots[1], states[-1][1])

    reset = StepCache()
    for index in range(prefix):
        model(tokens[:, index:index + 1], cache=reset)
    reset.slots = [None, None]
    restarted = model(tokens[:, :1], cache=reset)
    require_close("reset", restarted, stepped[0], tolerance=1e-6)

    fresh = StepCache()
    fresh_y = model(tokens[:, 1:2], cache=fresh)
    history_gap = float(mx.max(mx.abs(fresh_y - stepped[1])))
    if history_gap < 1e-3:
        raise SystemExit(f"{spec['name']} does not depend on recurrent/conv history ({history_gap})")

    projected = upstream["linear"](model.qkv_proj, tokens)
    manual_y, manual_state = manual_short_conv(
        model.qkv_conv.conv.weight,
        [projected[0, index] for index in range(spec["tokens"])],
    )
    conv_cache = StepCache()
    for index, (output, state) in enumerate(zip(manual_y, manual_state)):
        conv_only = model.qkv_conv(projected[:, index:index + 1], cache=conv_cache)
        require_close(f"manual conv output {index}", output[None, None], conv_only)
        require_close(f"manual conv state {index}", state[None], conv_cache.slots[0])
        require_close(f"attention conv state {index}", state[None], states[index][0])

    heads = spec["num_heads"]
    dim = spec["head_dim"]
    mixed = upstream["linear"](model.fbg_a_proj, tokens)
    features, _, gate_features = mx.split(mixed, (dim, dim + heads), axis=-1)
    pre_gate = upstream["linear"](model.f_b_proj, features).reshape(1, spec["tokens"], heads, dim)
    decay = upstream["compute_g_safe"](
        model.A_log.reshape(heads, 1),
        pre_gate,
        model.dt_bias.reshape(heads, dim),
        spec["lower_bound"],
    )
    output_gate = mx.sigmoid(upstream["linear"](model.g_b_proj, gate_features))
    mx.eval(decay, output_gate)
    decay_min = float(mx.min(decay))
    decay_max = float(mx.max(decay))
    output_gate_min = float(mx.min(output_gate))
    output_gate_max = float(mx.max(output_gate))
    first_head = decay[0, :, 0, :]
    span = float(mx.max(mx.max(first_head, axis=-1) - mx.min(first_head, axis=-1)))
    token_span = float(mx.max(mx.abs(decay[0, 0] - decay[0, 1])))
    if decay_min >= spec["decay_min_limit"] or decay_max <= spec["decay_max_limit"]:
        raise SystemExit(f"{spec['name']} safe gate is trivial: [{decay_min}, {decay_max}]")
    if span < spec["decay_span_limit"]:
        raise SystemExit(f"{spec['name']} decay does not vary inside the head ({span})")
    if token_span < 1e-4:
        raise SystemExit(f"{spec['name']} decay does not change across tokens")
    if output_gate_min >= spec["gate_min_limit"] or output_gate_max <= spec["gate_max_limit"]:
        raise SystemExit(
            f"{spec['name']} output gate is trivial: [{output_gate_min}, {output_gate_max}]"
        )
    if float(mx.max(mx.abs(states[-1][1]))) < 1e-4:
        raise SystemExit(f"{spec['name']} recurrent state stayed near zero")

    rows = []
    for index in range(spec["tokens"]):
        output = stepped[index][0, 0]
        convolution = states[index][0][0]
        recurrent = states[index][1][0]
        mx.eval(output, convolution, recurrent)
        output_list = flatten_f32(output)
        convolution_list = flatten_f32(convolution)
        recurrent_list = flatten_f32(recurrent)
        assert_f32_roundtrip(output_list, output)
        assert_f32_roundtrip(convolution_list, convolution)
        assert_f32_roundtrip(recurrent_list, recurrent)
        rows.append({
            "output": output_list,
            "convolution": convolution_list,
            "recurrent": recurrent_list,
        })
    fresh_output = flatten_f32(fresh_y[0, 0])
    fresh_convolution = flatten_f32(fresh.slots[0][0])
    fresh_recurrent = flatten_f32(fresh.slots[1][0])
    assert_f32_roundtrip(fresh_output, fresh_y[0, 0])
    assert_f32_roundtrip(fresh_convolution, fresh.slots[0][0])
    assert_f32_roundtrip(fresh_recurrent, fresh.slots[1][0])

    weights = {
        "qkv_projection": flatten_f32(model.qkv_proj.weight),
        "qkv_convolution": flatten_f32(model.qkv_conv.conv.weight),
        "fbg_a_projection": flatten_f32(model.fbg_a_proj.weight),
        "f_b_projection": flatten_f32(model.f_b_proj.weight),
        "g_b_projection": flatten_f32(model.g_b_proj.weight),
        "a_log": flatten_f32(model.A_log),
        "dt_bias": flatten_f32(model.dt_bias),
        "output_norm": flatten_f32(model.o_norm.weight),
        "output_projection": flatten_f32(model.o_proj.weight),
    }
    print(
        f"{spec['name']}: decay [{decay_min:.6f}, {decay_max:.6f}] "
        f"span {span:.6f} output-gate [{output_gate_min:.6f}, {output_gate_max:.6f}] "
        f"history gap {history_gap:.6f}"
    )
    return {
        "name": spec["name"],
        "hidden_size": spec["hidden_size"],
        "num_heads": spec["num_heads"],
        "head_dim": spec["head_dim"],
        "conv_kernel_size": spec["conv_kernel_size"],
        "lower_bound": spec["lower_bound"],
        "rms_norm_eps": spec["rms_norm_eps"],
        "prefix_length": prefix,
        "weights": weights,
        "tokens": [flatten_f32(tokens[0, index]) for index in range(spec["tokens"])],
        "steps": rows,
        "token1_fresh": {
            "output": fresh_output,
            "convolution": fresh_convolution,
            "recurrent": fresh_recurrent,
        },
        "decay_min": decay_min,
        "decay_max": decay_max,
        "decay_span_within_head": span,
        "output_gate_min": output_gate_min,
        "output_gate_max": output_gate_max,
        "history_gap": history_gap,
    }


def main():
    if not (REF / "language.py").is_file() or not (REF / "gated_delta.py").is_file():
        raise SystemExit(f"pinned sources not found at {REF}")
    try:
        import mlx.core as mx
    except ImportError as error:
        raise SystemExit(f"mlx is required ({error}); use ../glm-reference/.venv/bin/python")
    if mx.__version__ != "0.32.2":
        raise SystemExit(f"expected MLX 0.32.2, found {mx.__version__}")
    mx.set_default_device(mx.cpu)
    upstream = load_upstream()
    cases = [
        build_case(upstream, {
            "name": "vector-safe-gate",
            "hidden_size": 8,
            "num_heads": 2,
            "head_dim": 4,
            "conv_kernel_size": 4,
            "lower_bound": -5.0,
            "rms_norm_eps": 1e-5,
            "tokens": 5,
            "prefix_length": 2,
            "salt": 3,
            "qkv_scale": 0.18,
            "conv_scale": 0.35,
            "fbg_scale": 0.22,
            "fb_scale": 0.28,
            "gb_scale": 2.4,
            "norm_scale": 0.45,
            "out_scale": 0.3,
            "token_scale": 0.85,
            "a_log": [0.0, -0.3],
            "dt_bias": [-2.5, -1.0, 0.4, 2.2, -1.8, -0.2, 1.1, 2.8],
            "decay_min_limit": 0.05,
            "decay_max_limit": 0.6,
            "decay_span_limit": 0.4,
            "gate_min_limit": 0.25,
            "gate_max_limit": 0.75,
        }),
        build_case(upstream, {
            "name": "kernel-one-safe-gate",
            "hidden_size": 6,
            "num_heads": 1,
            "head_dim": 3,
            "conv_kernel_size": 1,
            "lower_bound": -1.5,
            "rms_norm_eps": 1e-5,
            "tokens": 4,
            "prefix_length": 2,
            "salt": 11,
            "qkv_scale": 0.2,
            "conv_scale": 0.4,
            "fbg_scale": 0.3,
            "fb_scale": 0.35,
            "gb_scale": 3.4,
            "norm_scale": 0.35,
            "out_scale": 0.28,
            "token_scale": 0.7,
            "a_log": [0.25],
            "dt_bias": [-3.0, 0.15, 2.6],
            "decay_min_limit": 0.4,
            "decay_max_limit": 0.8,
            "decay_span_limit": 0.35,
            "gate_min_limit": 0.3,
            "gate_max_limit": 0.7,
        }),
    ]
    payload = {
        "scope": "One GLM linear-attention block on the safe-gate ops path. No hyper-connections, sparse attention, or full-model inference.",
        "revision": REVISION,
        "mlx": mx.__version__,
        "device": "cpu",
        "use_kernel": False,
        "language_sha256": upstream["language_sha256"],
        "gated_delta_sha256": upstream["gated_delta_sha256"],
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
