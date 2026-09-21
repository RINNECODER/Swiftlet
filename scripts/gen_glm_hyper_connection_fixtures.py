"""Generate deterministic GLM hyper-connection oracles with MLX, not Swift.

The numeric results come from the pure ops in the pinned mlx-vlm
hyper_connection.py (CPU, FP32). The Metal sinkhorn/collapse kernel is not
used: its fast-exp and HC=4 specializations are not the reference math.
Helper imports in that module are unavailable here, so the pure functions are
AST-extracted and executed directly.
"""
import ast
import hashlib
import json
from pathlib import Path
from typing import Tuple

import mlx.core as mx

REVISION = "a74c7de90a344a2c2c7334acb4e48b57a40480e2"
SOURCE = Path(__file__).resolve().parent.parent.parent / "glm-reference" / "hyper_connection.py"
OUT = Path(__file__).resolve().parent.parent / "fixtures" / "glm-hyper-connection"
OPS = ("_hc_split_sinkhorn_ops", "_hc_ops", "_hc_expand_op")


def load_ops():
    source = SOURCE.read_bytes()
    tree = ast.parse(source)
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in OPS:
            node.decorator_list = []
            functions.append(node)
    found = [node.name for node in functions]
    if found != list(OPS):
        raise SystemExit(f"expected ops {OPS}, found {found} in {SOURCE}")
    namespace = {"mx": mx, "Tuple": Tuple}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return source, namespace


def as_floats(array):
    # Row-major flat layout, matching the Swift arrays.
    array = mx.array(array, dtype=mx.float32).reshape((-1,))
    mx.eval(array)
    return array.tolist()


def max_abs(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def sums(comb):
    rows = mx.sum(comb, axis=-1)
    cols = mx.sum(comb, axis=-2)
    return {
        "max_abs_row_sum_error": float(mx.max(mx.abs(rows - 1))),
        "max_abs_col_sum_error": float(mx.max(mx.abs(cols - 1))),
        "min_row_sum": float(mx.min(rows)),
        "max_row_sum": float(mx.max(rows)),
        "min_col_sum": float(mx.min(cols)),
        "max_col_sum": float(mx.max(cols)),
        "min_entry": float(mx.min(comb)),
        "max_entry": float(mx.max(comb)),
    }


def project(hc_ops, residual, fn, base, scale, hc_mult, iterations, hc_eps, rms_eps):
    y = residual.astype(mx.float32)
    flat = y.flatten(-2)
    # Same weightless RMS as HyperConnection.__call__, checked against the
    # documented mean-square formula so a silent MLX definition change fails here.
    normalized = mx.fast.rms_norm(flat, None, rms_eps)
    mean_square = mx.mean(mx.square(flat), axis=-1, keepdims=True)
    manual = flat * mx.rsqrt(mean_square + rms_eps)
    if max_abs(normalized, manual) > 1e-6:
        raise SystemExit(f"rms_norm diverged from rsqrt(mean(square)+eps): {max_abs(normalized, manual)}")
    mixes = normalized @ fn.T
    collapsed, post, comb = hc_ops(residual, y, mixes, scale, base, hc_mult, iterations, hc_eps)
    return mixes, collapsed, post, comb


def build_case(ops, name, *, batch, tokens, hidden, hc_mult, iterations, hc_eps, rms_eps, seed, zero_token=None):
    mx.random.seed(seed)
    width = hc_mult * hidden
    mix = (2 + hc_mult) * hc_mult
    fn = (mx.random.normal((mix, width)) * 0.2).astype(mx.float32)
    base = (mx.random.normal((mix,)) * 0.45).astype(mx.float32)
    # Distinct, non-unit scales. Base stays unscaled, matching the ops.
    scale = mx.array([0.85, 1.45, 0.55] if hc_mult != 2 else [1.25, 0.35, 1.7], dtype=mx.float32)
    if hc_mult == 3:
        scale = mx.array([0.6, 1.1, -0.8], dtype=mx.float32)
    stream_scale = mx.array([0.25, -1.6, 0.75, 2.3, -0.45][:hc_mult], dtype=mx.float32)
    stream_shift = mx.array([0.15, -0.55, 0.9, -1.25, 0.4][:hc_mult], dtype=mx.float32)
    residual = mx.random.normal((batch, tokens, hc_mult, hidden)).astype(mx.float32)
    residual = residual * stream_scale.reshape(1, 1, hc_mult, 1) + stream_shift.reshape(1, 1, hc_mult, 1)
    if zero_token is not None:
        values = residual.tolist()
        b, t = zero_token
        values[b][t] = [[0.0] * hidden for _ in range(hc_mult)]
        residual = mx.array(values, dtype=mx.float32)
    sublayer = (mx.random.normal((batch, tokens, hidden)) * 0.7 + 0.2).astype(mx.float32)
    mixes, collapsed, post, comb = project(
        ops["_hc_ops"], residual, fn, base, scale, hc_mult, iterations, hc_eps, rms_eps
    )
    direct_pre, direct_post, direct_comb = ops["_hc_split_sinkhorn_ops"](
        mixes, scale, base, hc_mult, iterations, hc_eps
    )
    if max_abs(direct_pre, mx.sigmoid(mixes[..., :hc_mult] * scale[0] + base[:hc_mult]) + hc_eps) > 1e-6:
        raise SystemExit("pre gate does not match sigmoid(mix * scale + base) + eps")
    expanded = ops["_hc_expand_op"](sublayer, residual, post, comb)
    mx.eval(mixes, collapsed, post, comb, expanded, direct_pre, direct_post, direct_comb)
    pre_gap = max_abs(direct_pre, mx.sigmoid(mixes[..., :hc_mult] * scale[0] + base[:hc_mult]) + hc_eps)
    if max(pre_gap, max_abs(direct_post, post), max_abs(direct_comb, comb)) > 1e-6:
        raise SystemExit("split sinkhorn ops diverged from _hc_ops")
    # Token-wise calls have no cross-token state. Fail if MLX disagrees.
    for token in range(tokens):
        part = residual[:, token:token + 1]
        part_mixes, part_collapsed, part_post, part_comb = project(
            ops["_hc_ops"], part, fn, base, scale, hc_mult, iterations, hc_eps, rms_eps
        )
        part_expanded = ops["_hc_expand_op"](sublayer[:, token:token + 1], part, part_post, part_comb)
        mx.eval(part_mixes, part_collapsed, part_expanded)
        gaps = (
            max_abs(part_mixes, mixes[:, token:token + 1]),
            max_abs(part_collapsed, collapsed[:, token:token + 1]),
            max_abs(part_expanded, expanded[:, token:token + 1]),
        )
        # Batched and single-token matmuls are the same math. MLX can still
        # differ by a couple of ulps because the reduction width changes.
        if max(gaps) > 1e-6:
            raise SystemExit(f"{name}: token {token} diverged from the batched op by mixes/collapsed/expanded {gaps}")
    if hc_mult > 1:
        stream_gap = float(mx.max(mx.abs(residual[:, :, 0, :] - residual[:, :, 1, :])))
        if stream_gap < 1e-2:
            raise SystemExit(f"{name}: residual streams are too similar ({stream_gap})")
    if zero_token is not None:
        b, t = zero_token
        start = (b * tokens + t) * hidden
        collapsed_flat = collapsed.reshape((-1,))
        if float(mx.max(mx.abs(collapsed_flat[start:start + hidden]))) != 0:
            raise SystemExit("zero residual token did not collapse to zero")
    # A 2x2 doubly stochastic matrix is symmetric, so hc_mult >= 3 is the
    # smallest width whose Sinkhorn mix can expose a transposed combination.
    if hc_mult >= 3:
        post_term = post[..., None] * sublayer[:, :, None, :].astype(mx.float32)
        transposed = post_term + mx.matmul(comb, residual.astype(mx.float32))
        gap = max_abs(transposed, expanded)
        if gap < 1e-2:
            raise SystemExit(f"{name}: transposed combination is too close to the reference expand ({gap})")
    spec = {
        "name": name,
        "batch": batch,
        "tokens": tokens,
        "hidden_size": hidden,
        "hc_mult": hc_mult,
        "hc_eps": hc_eps,
        "sinkhorn_iters": iterations,
        "rms_norm_eps": rms_eps,
        "seed": seed,
        "fn": as_floats(fn),
        "base": as_floats(base),
        "scale": as_floats(scale),
        "residual": as_floats(residual),
        "sublayer": as_floats(sublayer),
        "mixes": as_floats(mixes),
        "pre": as_floats(direct_pre),
        "post": as_floats(post),
        "comb": as_floats(comb),
        "collapsed": as_floats(collapsed),
        "expanded": as_floats(expanded),
        "normalization": sums(comb),
    }
    if zero_token is not None:
        spec["zero_token"] = {"batch": zero_token[0], "token": zero_token[1]}
    print(name, "comb", spec["normalization"])
    return spec, residual, fn, base, scale, sublayer


def iteration_sweep(ops, residual, fn, base, scale, sublayer, hc_mult, hc_eps, rms_eps):
    outputs = []
    collapsed_ref = None
    pre_ref = None
    previous = None
    for iterations in (0, 1, 2, 20):
        mixes, collapsed, post, comb = project(
            ops["_hc_ops"], residual, fn, base, scale, hc_mult, iterations, hc_eps, rms_eps
        )
        expanded = ops["_hc_expand_op"](sublayer, residual, post, comb)
        mx.eval(mixes, collapsed, post, comb, expanded)
        if collapsed_ref is None:
            collapsed_ref = collapsed
            pre_ref = mixes
        elif max_abs(collapsed, collapsed_ref) != 0 or max_abs(mixes, pre_ref) != 0:
            raise SystemExit("sinkhorn iterations changed collapse or mixes; ops contract broke")
        if previous is not None:
            gap = max_abs(comb, previous)
            expect_equal = iterations == 1  # max(iters - 1, 0) is 0 for both 0 and 1
            if expect_equal and gap != 0:
                raise SystemExit(f"iters {iterations} should match the previous comb, gap {gap}")
            if not expect_equal and gap < 1e-4:
                raise SystemExit(f"iters {iterations} did not move the comb, gap {gap}")
        previous = comb
        outputs.append({
            "sinkhorn_iters": iterations,
            "comb": as_floats(comb),
            "expanded": as_floats(expanded),
            "collapsed": as_floats(collapsed),
            "normalization": sums(comb),
        })
        print("sweep", iterations, outputs[-1]["normalization"])
    return outputs


def sequential_case(ops):
    batch, tokens, hidden, hc_mult = 1, 3, 6, 4
    iterations, hc_eps, rms_eps = 8, 1e-6, 1e-5
    mx.random.seed(91)
    width = hc_mult * hidden
    mix = (2 + hc_mult) * hc_mult
    residual = mx.random.normal((batch, tokens, hc_mult, hidden)).astype(mx.float32)
    residual = residual * mx.array([0.3, 1.8, -0.7, 2.1], dtype=mx.float32).reshape(1, 1, 4, 1)
    residual = residual + mx.array([-0.2, 0.45, -1.1, 0.8], dtype=mx.float32).reshape(1, 1, 4, 1)
    branches = {}
    current = residual
    for index, branch in enumerate(("attention", "feed_forward")):
        mx.random.seed(110 + index)
        fn = (mx.random.normal((mix, width)) * 0.18).astype(mx.float32)
        base = (mx.random.normal((mix,)) * 0.5).astype(mx.float32)
        scale = mx.array([0.7, 1.35, 0.4] if index == 0 else [1.15, 0.8, 1.55], dtype=mx.float32)
        sublayer = (mx.random.normal((batch, tokens, hidden)) * 0.55 + (0.1 if index == 0 else -0.25)).astype(mx.float32)
        mixes, collapsed, post, comb = project(
            ops["_hc_ops"], current, fn, base, scale, hc_mult, iterations, hc_eps, rms_eps
        )
        pre, _, _ = ops["_hc_split_sinkhorn_ops"](mixes, scale, base, hc_mult, iterations, hc_eps)
        expanded = ops["_hc_expand_op"](sublayer, current, post, comb)
        mx.eval(expanded, collapsed, comb, mixes, post)
        # One token at a time, carrying the expanded stream into the next branch.
        carried = []
        for token in range(tokens):
            part = current[:, token:token + 1]
            _, _, part_post, part_comb = project(
                ops["_hc_ops"], part, fn, base, scale, hc_mult, iterations, hc_eps, rms_eps
            )
            carried.append(ops["_hc_expand_op"](sublayer[:, token:token + 1], part, part_post, part_comb))
        carried = mx.concatenate(carried, axis=1)
        mx.eval(carried)
        chain_gap = max_abs(carried, expanded)
        if chain_gap > 1e-6:
            raise SystemExit(f"sequential {branch} token chain diverged by {chain_gap}")
        branches[branch] = {
            "fn": as_floats(fn),
            "base": as_floats(base),
            "scale": as_floats(scale),
            "sublayer": as_floats(sublayer),
            "mixes": as_floats(mixes),
            "pre": as_floats(pre),
            "post": as_floats(post),
            "comb": as_floats(comb),
            "collapsed": as_floats(collapsed),
            "expanded": as_floats(expanded),
            "normalization": sums(comb),
        }
        current = expanded
    if max_abs(current, residual) < 1e-2:
        raise SystemExit("sequential chain did not move the residual")
    print("sequential attention", branches["attention"]["normalization"])
    print("sequential ffn", branches["feed_forward"]["normalization"])
    return {
        "batch": batch,
        "tokens": tokens,
        "hidden_size": hidden,
        "hc_mult": hc_mult,
        "hc_eps": hc_eps,
        "sinkhorn_iters": iterations,
        "rms_norm_eps": rms_eps,
        "residual": as_floats(residual),
        "attention": branches["attention"],
        "feed_forward": branches["feed_forward"],
    }


def main():
    if mx.__version__ != "0.32.2":
        raise SystemExit(f"expected MLX 0.32.2, found {mx.__version__}")
    mx.set_default_device(mx.cpu)
    source, ops = load_ops()
    OUT.mkdir(parents=True, exist_ok=True)
    default, residual, fn, base, scale, sublayer = build_case(
        ops, "glm_default", batch=2, tokens=3, hidden=8, hc_mult=4,
        iterations=20, hc_eps=1e-6, rms_eps=1e-5, seed=53, zero_token=(0, 1),
    )
    others = [
        build_case(ops, "hc2", batch=1, tokens=4, hidden=6, hc_mult=2,
                   iterations=5, hc_eps=1e-4, rms_eps=1e-3, seed=61)[0],
        build_case(ops, "hc3", batch=2, tokens=2, hidden=5, hc_mult=3,
                   iterations=1, hc_eps=1e-6, rms_eps=1e-5, seed=67)[0],
        build_case(ops, "hc1", batch=1, tokens=2, hidden=4, hc_mult=1,
                   iterations=3, hc_eps=1e-6, rms_eps=1e-5, seed=71)[0],
    ]
    payload = {
        "format": "glm-hyper-connection-v1",
        "revision": REVISION,
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "mlx": mx.__version__,
        "device": "cpu",
        "kernel_used": False,
        "extracted_ops": list(OPS),
        "scope": "Manifold-constrained hyper-connection ops only. No attention, MoE, HyperHead, or full model.",
        "cases": [default, *others],
        "iteration_sweep": {
            "batch": default["batch"],
            "tokens": default["tokens"],
            "hidden_size": default["hidden_size"],
            "hc_mult": default["hc_mult"],
            "hc_eps": default["hc_eps"],
            "rms_norm_eps": default["rms_norm_eps"],
            "fn": default["fn"],
            "base": default["base"],
            "scale": default["scale"],
            "residual": default["residual"],
            "sublayer": default["sublayer"],
            "mixes": default["mixes"],
            "pre": default["pre"],
            "post": default["post"],
            "iterations": iteration_sweep(
                ops, residual, fn, base, scale, sublayer, 4, 1e-6, 1e-5
            ),
        },
        "sequential": sequential_case(ops),
    }
    (OUT / "reference.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUT / 'reference.json'}")


if __name__ == "__main__":
    main()
