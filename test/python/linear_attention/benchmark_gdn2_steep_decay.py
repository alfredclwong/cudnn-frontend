# SPDX-FileCopyrightText: Copyright (c) 2026 Alfred Wong
# SPDX-License-Identifier: Apache-2.0

"""Synthetic GDN-2 forward/backward benchmark; no production data required.

Run this identical script in separate processes for base/patch/patch/base on
one GPU. Check the recorded import path and source hash for each arm. CUDA
 events measure the public autograd call, including GPU idle gaps while the
host enqueues work, not pure kernel active time. Warmup excludes compilation.
Finiteness checks do not replace the FP64 regression tests.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F

import cudnn
from cudnn.linear_attention.ops.gdn2 import gated_delta_net_v2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7082)
    args = parser.parse_args()
    if min(args.tokens, args.heads, args.warmup, args.repeats) < 1:
        parser.error("tokens, heads, warmup and repeats must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.manual_seed(args.seed)
    shape = (args.tokens, args.heads, 128)
    q = F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16().requires_grad_()
    k = F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16().requires_grad_()
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16).requires_grad_()
    g = torch.full(shape, -0.02, device="cuda", dtype=torch.float32).requires_grad_()
    b = torch.full(shape, 0.5, device="cuda", dtype=torch.bfloat16).requires_grad_()
    w = torch.full(shape, 0.5, device="cuda", dtype=torch.bfloat16).requires_grad_()
    offsets = torch.tensor([0, args.tokens], device="cuda", dtype=torch.int32)
    grad_output = torch.randn_like(v)
    inputs = (q, k, v, g, b, w)

    def call():
        output, _ = gated_delta_net_v2(
            q,
            k,
            v,
            g,
            b,
            w,
            offsets,
            scale=128**-0.5,
            use_qk_l2norm_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
            allow_neg_eigval=False,
            safe_gate=False,
            gate_domain="log",
            plan_name="gdn2_frost",
        )
        return output, torch.autograd.grad(output, inputs, grad_output)

    results = []
    for count in (0, 1, 4, 16, 64, 128):
        with torch.no_grad():
            g.fill_(-0.02)
            g[:, :, :count] = -6.0
        for _ in range(args.warmup):
            output, grads = call()
        torch.cuda.synchronize()
        samples = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(args.repeats):
            start.record()
            output, grads = call()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
            if not torch.isfinite(output).all() or not all(torch.isfinite(grad).all() for grad in grads):
                raise RuntimeError(f"Nonfinite result with {count} steep channels/head")
        row = {"steep_channels_per_head": count, "median_ms": statistics.median(samples), "samples_ms": samples}
        results.append(row)
        print(json.dumps(row), flush=True)

    helper = Path(cudnn.__file__).parent / "linear_attention/frost/common/steep_decay.py"
    record = {
        "label": args.label,
        "shape": shape,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "cudnn_backend": cudnn.backend_version(),
        "cudnn_file": cudnn.__file__,
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "results": results,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
