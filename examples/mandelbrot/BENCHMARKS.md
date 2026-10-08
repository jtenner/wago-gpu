# Mandelbrot benchmark results

Measured on 2026-10-08 using an NVIDIA RTX 4060 Laptop GPU, Vulkan,
driver 550.163.01, and an AMD Ryzen 7 8845HS CPU. The host ran Debian 13,
Linux 6.12, and Go 1.27.1. Each image used 32 F32 iterations.

The direct Wago CPU program was faster than the GPU at every tested size.
The GPU was much faster than the scalar buffer CPU fallback. These paths
produce the same image, but the direct CPU stops work for escaped pixels.
The buffer paths update every pixel on every pass and make extra transfers.

## End-to-end time

Times are milliseconds per complete WASI command, including instance creation,
guest setup, image generation, WASI output to `io.Discard`, checks, and instance
cleanup. The image byte comparison is outside the timer.

`SetupAndRender` also includes runtime, device, module, and pipeline setup plus
cleanup. It uses fresh objects in the same process; driver caches can be warm.
`ReuseModule` excludes that setup and retains the module and device. It still
creates a new instance and four new buffers per image. It is not a resident
kernel-only measurement.

CPU/GPU entries are the medians of three benchmark samples, each averaging
three timed commands. Fallback entries have one timed command each because
that path is slow. Do not treat the fallback values as a stable statistical
estimate.

### Setup included

| Image | Direct CPU ms | Buffer CPU fallback ms | GPU ms | GPU / direct CPU |
| --- | ---: | ---: | ---: | ---: |
| 96x64 | 1.092 | 859.969 | 95.401 | 87.4x |
| 256x192 | 2.048 | 6939.405 | 98.999 | 48.3x |
| 512x512 | 6.686 | 37767.245 | 128.531 | 19.2x |

### Module and device reused

| Image | Direct CPU ms | Buffer CPU fallback ms | GPU ms | GPU / direct CPU |
| --- | ---: | ---: | ---: | ---: |
| 96x64 | 0.173 | 859.617 | 5.617 | 32.5x |
| 256x192 | 1.063 | 7566.327 | 9.390 | 8.8x |
| 512x512 | 5.452 | 38139.107 | 33.182 | 6.1x |

## First-run costs

These are separate measurements from the reused-module cases. First render
includes buffer creation and initial uploads, but excludes the preceding
runtime/device/module setup. Pipeline creation occurs during that first render.
Wasm compilation includes the source hook; translation is part of that time.
The native pipeline measurement can omit driver work deferred to submission.
Do not add these overlapping columns together.

| Image | GPU device init ms | Wasm compile ms | Wasm-to-WGSL ms | Pipeline ms | First GPU render ms | Repeated GPU render ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 96x64 | 55.900 | 1.510 | 0.019 | 0.258 | 16.320 | 5.617 |
| 256x192 | 55.470 | 1.586 | 0.018 | 0.277 | 20.520 | 9.390 |
| 512x512 | 60.280 | 1.989 | 0.022 | 0.299 | 45.560 | 33.182 |

## Transfer and memory costs

The following GPU bytes are per complete reused-module command. Both orbit
buffers are read back after each of the 32 passes, and both outputs are seeded
from their old device contents. Fresh instances require initial uploads.
The plugin counters include these transfers; parameter uploads are separate
and are not included in this table. MiB means 1,048,576 bytes.

Go allocation counts cover the timed host path. They exclude native driver
allocations, mapped Wasm memory, and guest heap use. Tracked plugin peaks cover
plugin-owned buffer storage and exclude driver-private memory.

| Image | GPU upload MiB | GPU readback MiB | GPU seed copies MiB | Plugin tracked peak MiB | CPU Go B/op | GPU Go B/op | CPU allocs/op | GPU allocs/op |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96x64 | 0.0938 | 1.5000 | 1.5000 | 0.2344 | 32,034 | 267,893 | 427 | 2,880 |
| 256x192 | 0.7500 | 12.0000 | 12.0000 | 1.8750 | 32,040 | 956,152 | 429 | 2,884 |
| 512x512 | 4.0000 | 64.0000 | 64.0000 | 10.0000 | 32,040 | 4,364,237 | 429 | 2,887 |

Stage profiling was disabled for these timing runs. This avoids its extra
queue waits. Device timestamps are unavailable, so no isolated GPU execution
or transfer-duration claim is made. End-to-end time and transfer byte counts
are measured. The earlier array-kernel report contains separate stage-profiled
measurements; those are different workloads.

## Fresh-process default command

The executable was built before timing. Each of the following entries is the
median of three fresh processes at 96x64 pixels and 32 iterations. Wall time
includes process launch, all setup, output to a temporary file, and cleanup.
Maximum RSS is the process resident-set high-water mark reported by Linux `wait4`;
it is not GPU VRAM use.

| Mode | Process wall ms | Max RSS MiB |
| --- | ---: | ---: |
| cpu | 5.052 | 17.61 |
| fallback | 1010.114 | 17.61 |
| gpu | 147.220 | 181.21 |

## Correctness and limits

All image comparisons in these benchmark runs reported zero differing pixels.
Every GPU command had to report 32 GPU passes. Every buffer command had to free
its handles. The separate CPU and hardware race tests passed.

Floating-point arithmetic remains relaxed on the GPU. This image agreement
does not prove bit agreement for other grids or devices. Clocks were not fixed,
and there was no thermal control or statistical confidence interval. A short
CPU/hardware race-test run occurred while the fallback benchmark process was
active; it was not an isolated-machine run. The large fallback values have only
one timed sample. No GPU crossover was found within the supported image sizes.

The largest GPU image requires 64 MiB of readback and 64 MiB of device-local
seed copies. Avoiding the per-pass readbacks, keeping escape tests on the GPU,
and omitting seed copies when a safe proof permits it are useful next work.
Those changes are not part of these measurements.

## Evidence and rerun commands

- [Benchmark source](benchmark_test.go) and [commands](README.md#benchmarks).
- [Fresh-process measurement helper](bench_processes.py), using Linux `wait4`.
- [CPU/GPU raw samples](../../results/mandelbrot-bench.txt).
- [Fallback raw samples](../../results/mandelbrot-bench-fallback.txt).
- [Fresh-process records](../../results/mandelbrot-cold-processes.json).
- [Environment and file hashes](../../results/mandelbrot-bench-environment.txt).
- [CPU and hardware race-test log](../../results/mandelbrot-bench-tests.txt).
