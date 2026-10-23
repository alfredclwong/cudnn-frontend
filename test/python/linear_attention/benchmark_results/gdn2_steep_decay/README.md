# Standalone upstream results

One B300, public `nvcr.io/nvidia/pytorch:25.10-py3` container, seed 7082, ABBA order.
See [the reproduction instructions](../../benchmark_gdn2_steep_decay.md) for the full environment and commands.

| Tokens | Steep channels | Base GPU busy (µs) | Patch GPU busy (µs) | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0 | 32.47 | 32.50 | -0.08% |
| 16 | 1 | 107.77 | 66.22 | 38.56% |
| 16 | 64 | 208.22 | 125.11 | 39.91% |
| 256 | 0 | 98.09 | 98.15 | -0.07% |
| 256 | 1 | 431.77 | 265.62 | 38.48% |
| 256 | 64 | 890.90 | 559.92 | 37.15% |

Values are the mean of two per-arm medians. Each median has five profiler samples of ten forward/backward calls.
GPU busy time is the union of GPU event intervals, with idle gaps excluded and overlaps counted once.
Profiler overhead is present. Eager CUDA-event timings include host enqueue gaps and vary substantially;
these results do not establish an end-to-end throughput improvement. All eager samples remain in the raw records.

## Correctness and path activation

All 24 cases passed the 1% FP64 relative Frobenius tolerance for the output and all six input gradients.
The worst relative error was 0.0054148377 (0.5415%).
Every mild case left the backward exact-path scheduler counter at zero.
Every steep case advanced it to 2 for 16 tokens and 5 for 256 tokens, confirming exact correction work.
The counter is not a tile or steep-channel count. Reads occur outside timing.

## Provenance and scope

- Upstream base: `f2aa360e5a08436651d158cfad26b05b7db5ab92`.
- Kernel patch: `ab9a46584daa6f3e134c7b1bf4b1074ca453710e`.
- The two runtime copies use the same built wheel and differ only in `cudnn/linear_attention/frost/common/steep_decay.py` (1,175 runtime files checked).
- `provenance.json` records the public image, wheel hash, and helper hashes.
- `base-A1.json`, `patch-B1.json`, `patch-B2.json`, and `base-A2.json` contain all samples, FP64 errors, scheduler observations, kernel event totals, software versions, and actual import paths.
- `summary.json` is generated from those four files by `summarize_gdn2_steep_decay.py`.
- Recorded harness SHA256: `dcc4d3ff87475457b5f368e3e3a89bc961fcbd11cf79a1225924f5687cc0fd0e`.

These are small synthetic mechanism checks. They do not establish model convergence, application throughput,
other architectures, or correctness across the full upstream regression suite. The PR remains in draft for broader qualification.
