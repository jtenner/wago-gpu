# Larger images and scratch-buffer performance

These are the earlier scratch-buffer measurements. The
[follow-up report](PERFORMANCE_FOLLOWUP.md) contains the latest changes and results.

The example now defaults to **1920×1080 pixels and 64 iterations**. It accepts
up to 4096 pixels on each axis, 4,194,304 total pixels, and 128 iterations.
This includes 2560×1440. Host and guest share the limits. The plugin's existing
256 MiB instance cap remains unchanged. The preview is actual full-HD WASI
output, not an enlarged small image.

## Changes

- Retain completed scratch for all outputs, up to the eight-slot ABI limit.
  Previously, the second Mandelbrot output allocated and destroyed one GPU
  buffer on every pass.
- Keep one readback buffer and one 16-byte uniform block per instance. Reuse
  follows completed work and successful unmap. Failed or uncertain storage is
  retired and charged until cleanup.
- Retain one bounded Go conversion slice for narrow uploads. Active storage is
  removed from idle pools before other reservations. Idle capacity is reclaimed
  before a budget rejection, and growth overlap is charged.
- Use eight-entry dispatch arrays instead of growing slices, sorting, and an
  output map. Alias validation also uses bounded storage. Scalar imports omit
  unused queue/profile clocks; GPU-dependent timeouts still start at entry.
- Allocate guest arrays once. CPU fallback copies current inputs in bulk,
  performs the same F32 formulas in native Wasm memory, and sets outputs in bulk.
  The escape check uses the existing orbit arrays. This replaces 12 host calls
  per pixel per pass with a fixed number of calls per pass.

No element-count-dependent quadratic loop was found. All pixel/transfer loops
remain linear. Pipeline caching already existed. No Wago source or public
plugin ABI was changed.

## Measurements

Measured on the same NVIDIA RTX 4060 Laptop/Vulkan device, driver 550.163.01,
AMD Ryzen 7 8845HS, Debian 13/Linux 6.12, and Go 1.27.1. `GOMAXPROCS=1`,
`ProfileStages=false`, `-benchtime=3x`, `-count=3`, `-benchmem`; no race detector.
Each table value is the median of three samples, each averaging three renders.
These measurements use **32 iterations** for comparison with the earlier run.
The new default and preview use 64.

`ReuseModule` retains the module, device, and pipeline, but each command gets a
fresh WASI instance and fresh guest buffers. Time includes guest setup, the
render, WASI output to `io.Discard`, result checks, and instance cleanup. Image
comparison is outside timing. No GPU device timestamps are available.

### Current reused-module command times

| Image | Direct CPU ms | Bulk buffer CPU ms | GPU ms |
| --- | ---: | ---: | ---: |
| 512x512 | 5.483 | 27.060 | 31.437 |
| 1920x1080 | 43.623 | 235.632 | 215.152 |
| 2560x1440 | 73.169 | 466.488 | 367.325 |

The direct CPU remains faster at each measured size. It stops updating escaped
pixels; the buffer paths update every pixel on every pass. The large GPU path
continues to read back two orbit buffers and seed two output buffers per pass.

### GPU scratch comparison

Large-image before and after runs use the same image grids and benchmark
settings. Before already used the larger limits, but did not have the scratch
and bulk-runner changes.

| Image | GPU before ms | GPU after ms | Go allocations before/op | Go allocations after/op | Tracked peak before MiB | Tracked peak after MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1920x1080 | 228.052 | 215.152 | 2887 | 2480 | 79.10 | 87.01 |
| 2560x1440 | 401.087 | 367.325 | 2887 | 2480 | 140.60 | 154.70 |

Retaining safe scratch increases idle memory. It does not make it unbounded.
The largest measured peak remains below the 256 MiB cap. Every retained byte is
charged, and idle capacity can be evicted. Go B/op mostly comes from the four
fresh plugin CPU mirrors per command; scratch reuse does not remove that setup
allocation. Native driver allocations are outside Go allocation counters.

### CPU fallback comparison

The earlier scalar fallback had one timed sample per size in a CPU-only build.
The current bulk runner has three samples of three commands in the GPU-enabled
build with the GPU disabled. These are local comparisons, not controlled
confidence estimates.

| Image | Earlier scalar fallback ms | Current bulk fallback ms |
| --- | ---: | ---: |
| 96x64 | 859.617 | 0.986 |
| 256x192 | 7566.327 | 4.897 |
| 512x512 | 38139.107 | 27.060 |

## Correctness and numerical behavior

CPU and bulk-fallback images match exactly. At 32 iterations, GPU comparisons
reported five different pixels out of 2,073,600 at full HD, and thirteen out of
3,686,400 at 1440p. The counts are unchanged before and after the storage
changes. GPU arithmetic remains relaxed; the small image's earlier exact
agreement did not extend to these finer grids.

Full-HD and 1440p one-pass hardware tests compare all bytes exactly and exercise
large buffers. Additional fake-backend tests require allocation counts to stop
after the first two-output/readback pass, check active conversion under budget
pressure, growth accounting, failed unmap after CPU cache population, unchanged
guest bytes on failed transfer, queue-lock timeout, and exact cleanup. A mixed
WASI test runs one fake-GPU pass, injects a failure, and checks eleven bulk CPU
fallback passes against the direct CPU image.

The full CPU-only suite, real-GPU race suite, and cgo pointer checks passed.
Clocks were not fixed, and there was no thermal control. Small GPU timing
changes can include measurement noise. The large before/after benchmark
processes ran sequentially without other test processes started by this task.

The 64-iteration image checks are recorded separately:

| Image | CPU vs bulk fallback differing pixels | CPU vs GPU differing pixels |
| --- | ---: | ---: |
| 1920x1080 | 0 | 198 |
| 2560x1440 | 0 | 366 |

Wasmtime also produced the same default full-HD CPU file as Wago. All required
GPU runs reported 64 GPU passes and no fallback. The preview uses the CPU image.

## Original array workload checks

The original `2*x` and `x*x+1` benchmark ran again at 1,024; 16,384; 1,000,000;
and 10,000,000 elements. Each case had one first GPU render and one repeated
fresh-input GPU render, plus resident and prefix checks. Every output was
verified. This run uses stage profiling and is separate from Mandelbrot timing.

| Kernel | Elements | Repeated GPU end-to-end ms | Direct CPU ms | Peak tracked MiB |
| --- | ---: | ---: | ---: | ---: |
| twice | 1,024 | 0.509 | 0.011 | 0.03 |
| twice | 16,384 | 0.362 | 0.020 | 0.44 |
| twice | 1,000,000 | 1.828 | 0.580 | 26.70 |
| twice | 10,000,000 | 89.843 | 7.510 | 228.88 |
| square | 1,024 | 0.273 | 0.010 | 0.03 |
| square | 16,384 | 0.298 | 0.023 | 0.44 |
| square | 1,000,000 | 1.653 | 0.785 | 26.70 |
| square | 10,000,000 | 89.429 | 7.794 | 228.88 |

## Evidence

- [Large before samples](../../results/mandelbrot-large-before.txt).
- [Large after samples](../../results/mandelbrot-large-after.txt).
- [Small after samples](../../results/mandelbrot-small-after.txt).
- [Large image hashes and commands](../../results/mandelbrot-large-images.json).
- [Original array benchmark JSON](../../results/perf-array-bench.jsonl).
- [CPU-only tests](../../results/perf-cpu-tests.txt), [real-GPU race tests](../../results/perf-hardware-tests.txt), and [cgo checks](../../results/perf-cgocheck-tests.txt).
- [Benchmark source and rerun commands](README.md#benchmarks).

Next work should reduce the remaining transfers by moving the escape check and
loop into the supported GPU subset. Seed omission needs a valid dataflow proof;
the current read-before-write kernels cannot skip it with the current binding
layout. Neither change is claimed here.
