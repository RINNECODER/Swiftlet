"""Generate small, deterministic GLM port oracles using MLX, not Swift code.

Run with Python + mlx. Downloads ONLY two reference functions from
pinned mlx-vlm source, never model weights. The source hash and versions are
recorded in the fixture metadata. Full GLM attention is outside this fixture.
"""
import ast
import hashlib
import json
from pathlib import Path
from typing import Tuple
from urllib.request import urlopen

import mlx.core as mx
import mlx.nn as nn

REVISION = "a74c7de90a344a2c2c7334acb4e48b57a40480e2"
SOURCE_URL = f"https://raw.githubusercontent.com/Blaizzy/mlx-vlm/{REVISION}/mlx_vlm/models/glm5_next/language.py"
OUT = Path(__file__).resolve().parent.parent / "fixtures" / "glm-next"


def main():
    mx.set_default_device(mx.cpu)
    mx.random.seed(53)
    source = urlopen(SOURCE_URL, timeout=30).read()
    tree = ast.parse(source)
    # Evaluate the upstream functions themselves, not copies of our Swift logic.
    names = {"_expert_select", "_limited_swiglu"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(functions) == len(names)
    namespace = {"mx": mx, "nn": nn, "Tuple": Tuple}
    exec(compile(ast.Module(body=functions, type_ignores=[]), SOURCE_URL, "exec"), namespace)
    select, activation = namespace["_expert_select"], namespace["_limited_swiglu"]
    OUT.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "glm5_next", "text_config": {
            "model_type": "glm5_next_text", "hidden_size": 64, "intermediate_size": 64,
            "moe_intermediate_size": 64, "num_hidden_layers": 2, "first_k_dense_replace": 1,
            "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2,
            "n_group": 2, "topk_group": 1, "norm_topk_prob": True,
            "routed_scaling_factor": 2.5, "swiglu_limit": 1.0,
            "linear_attn_config": {"num_heads": 2, "head_dim": 8, "short_conv_kernel_size": 4},
        }, "quantization": {"group_size": 32, "bits": 4},
    }
    stored, decoded = {}, {}

    def put(module, weights, bits):
        q, s, b = mx.quantize(weights, group_size=32, bits=bits)
        stored.update({f"{module}.weight": q, f"{module}.scales": s, f"{module}.biases": b})
        config["quantization"][module] = {"group_size": 32, "bits": bits}
        decoded[module] = mx.dequantize(q, s, b, group_size=32, bits=bits)

    def random(shape):
        return mx.random.normal(shape) * 0.15

    prefix = "language_model.model.layers.1.mlp"
    stored[prefix + ".gate.weight"] = random((8, 64))
    # Large bias forces expert selection to differ from raw sigmoid ranking.
    correction = mx.array([-0.5, 0.3, 0.1, 0.8, 0.9, -0.8, 0.5, -0.3])
    stored[prefix + ".gate.e_score_correction_bias"] = correction
    for projection, bits in [("gate_proj", 4), ("up_proj", 4), ("down_proj", 5)]:
        put(prefix + ".switch_mlp." + projection, random((8, 64, 64)), bits)
    for pfx in ["language_model.model.layers.0.mlp", prefix + ".shared_experts"]:
        put(pfx + ".gate_up_proj", random((128, 64)), 6)
        put(pfx + ".down_proj", random((64, 64)), 6)

    # Cover all MLX affine bit widths and each supported group size, including
    # fields crossing byte/word boundaries. BF16 scales exercise dtype handling.
    for bits in [2, 3, 4, 5, 6, 8]:
        for gs in [32, 64, 128]:
            module = f"quant{bits}g{gs}"
            w = random((3, 256)).astype(mx.bfloat16)
            q, s, b = mx.quantize(w, group_size=gs, bits=bits)
            stored.update({module + ".weight": q, module + ".scales": s, module + ".biases": b})
            config["quantization"][module] = {"group_size": gs, "bits": bits}
            decoded[module] = mx.dequantize(q, s.astype(mx.float32), b.astype(mx.float32), group_size=gs, bits=bits)
    x = random((3, 64)) * 4

    def mlp(pfx, value):
        gate, up = mx.split(value @ decoded[pfx + ".gate_up_proj"].T, 2, axis=-1)
        return activation(gate, up, 1.0) @ decoded[pfx + ".down_proj"].T

    logits = x @ stored[prefix + ".gate.weight"].T
    ids, weights = select(logits, correction, 2, 2, 1, 2.5, True)
    rows = []
    mx.eval(ids, weights)
    for i in range(x.shape[0]):
        result = mx.zeros((64,))
        for j, e in enumerate(ids[i].tolist()):
            gate = x[i] @ decoded[prefix + ".switch_mlp.gate_proj"][e].T
            up = x[i] @ decoded[prefix + ".switch_mlp.up_proj"][e].T
            expert = activation(gate, up, 1.0) @ decoded[prefix + ".switch_mlp.down_proj"][e].T
            result = result + weights[i, j] * expert
        rows.append(result + mlp(prefix + ".shared_experts", x[i]))
    oracle = {"input": x, "logits": logits, "ids": ids, "route_weights": weights,
              "sparse_output": mx.stack(rows), "dense_output": mlp("language_model.model.layers.0.mlp", x),
              **{k: v for k, v in decoded.items() if k.startswith("quant")}}
    # Routing-only cases include top-k=1 (must NOT normalize), ungrouped,
    # grouped with two retained groups, and disabled top-k normalization.
    routing = []
    for top_k, groups, kept, normalize in [(1, 1, 1, True), (3, 1, 1, True), (2, 2, 1, False), (3, 4, 2, True)]:
        ri, rw = select(logits, correction, top_k, groups, kept, 2.5, normalize)
        mx.eval(ri, rw)
        routing.append(dict(top_k=top_k, groups=groups, kept=kept, normalize=normalize, ids=ri.tolist(), weights=rw.tolist()))
    mx.eval(stored, oracle)
    # Split projections across shards so weight/scales/biases cannot be assumed
    # to live together. For split experts the quant overrides are renamed too.
    for layout in ["stacked", "split"]:
        tensors, cfg = dict(stored), json.loads(json.dumps(config))
        if layout == "split":
            for projection in ["gate_proj", "up_proj", "down_proj"]:
                old = prefix + ".switch_mlp." + projection
                spec = cfg["quantization"].pop(old)
                for suffix in ["weight", "scales", "biases"]:
                    values = tensors.pop(old + "." + suffix)
                    for expert in range(8):
                        module = prefix + f".experts.{expert}." + projection
                        tensors[module + "." + suffix] = values[expert]
                        cfg["quantization"][module] = spec
        directory = OUT / layout
        directory.mkdir(exist_ok=True)
        (directory / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        parts = [{}, {}, {}]
        for index, (name, value) in enumerate(sorted(tensors.items())):
            parts[index % 3][name] = value
        weight_map = {}
        for i, part in enumerate(parts):
            filename = f"model-{i + 1:05d}-of-00003.safetensors"
            mx.save_safetensors(str(directory / filename), part)
            weight_map.update({name: filename for name in part})
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}, indent=2) + "\n")
    mx.save_safetensors(str(OUT / "reference.safetensors"), oracle)
    (OUT / "reference.json").write_text(json.dumps({"source": SOURCE_URL,
        "source_sha256": hashlib.sha256(source).hexdigest(), "mlx": mx.__version__, "seed": 53,
        "scope": "Feed-forward only; no GLM attention or full-model generation", "routing_cases": routing,
    }, indent=2) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
