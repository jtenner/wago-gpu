# All example benchmark results

Measured on 2026-10-08. All runnable programs in `examples/` were measured.
The two commands in `cmd/` were also measured. GPU timings use real hardware.
All required GPU dispatches succeeded. No fallback is counted as GPU execution.

## Hardware and settings

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB, Vulkan, driver 550.163.01.
- CPU: AMD Ryzen 7 8845HS.
- System: Debian 13.7, Linux 6.12.111+deb13-amd64, Go 1.27.1, amd64.
- Wago: `v0.1.0-beta.12.0.20261007220511-7aa401f29a33`.
- WebGPU binding: `v1.36.0`, with the checked-in native safety patches.
- `GOMAXPROCS=1`. Timed programs ran one at a time. No race detector during timing.
- Clocks, temperature, and driver caches were not fixed. These results apply to this machine.

The implementation is based on commit `d8c2d7b`. This revision adds benchmark
code and corrects the dependency name in the demo's metadata. It does not change
kernel execution. [Machine settings](../results/examples-2026-10-08-settings.json)
and [sample summary](../results/examples-2026-10-08-summary.json) are saved.
The summary contains all three samples and their minimum, median, and maximum.

## Small examples

All times in this table are milliseconds per complete guest command.
Each command checks its result and execution path. Console printing is excluded.

**Full setup** creates and closes the runtime, device, module, and instance.
Driver caches can be warm. It does not include process launch or building Go code.
**Retained module** reuses the module and device. It creates a fresh instance,
runs the guest, checks the result, and closes the instance. It is not a
kernel-only timing. Start-section programs execute during instance creation.

Each value is the median of three samples. Full setup samples have three timed
commands each. Retained-module samples have 100 commands each. An untimed
checked command precedes each sample. CPU cases disable device creation.
This includes the all-CPU buffer case; its normal host flags use thresholds
without disabling the device. Mixed cases use thresholds for the selected CPU kernel.

| Program | Full setup CPU ms | Full setup GPU ms | Retained CPU ms | Retained GPU ms |
| --- | ---: | ---: | ---: | ---: |
| Start section; four F32 values | 0.197 | 81.810 | 0.010 | 0.086 |
| Two buffer kernels; four F32 values | 0.311 | 83.427 | 0.028 | 0.214 |
| TinyGo guest; four F32 values | 0.363 | 84.527 | 0.025 | 0.171 |
| F16 increment; one value | 0.296 | 91.045 | 0.019 | 0.174 |
| Memory64 transfer; one F32 value | 0.231 | N/A | 0.016 | N/A |
| GC-array transfer; one F32 value | 0.249 | N/A | 0.051 | N/A |

The storage transfer programs have no GPU dispatch. Their GPU entries are N/A.
All arithmetic results in these small programs matched their expected bits.
GPU setup costs dominate their full setup times.

The two buffer kernels also passed both mixed execution paths:

| First → second kernel | Full setup ms | Retained module ms |
| --- | ---: | ---: |
| GPU → CPU | 85.071 | 0.180 |
| CPU → GPU | 85.689 | 0.184 |

### Go allocation cost with the module retained

Bytes below are allocated Go bytes per command, not resident memory.
Native driver storage and much of the Wasm memory are excluded.

| Program | CPU KiB/command | CPU allocations | GPU KiB/command | GPU allocations |
| --- | ---: | ---: | ---: | ---: |
| Start section; four F32 values | 3.88 | 44 | 4.43 | 59 |
| Two buffer kernels; four F32 values | 12.43 | 154 | 15.92 | 225 |
| TinyGo guest; four F32 values | 11.57 | 152 | 13.87 | 203 |
| F16 increment; one value | 9.40 | 132 | 11.71 | 184 |
| Memory64 transfer; one F32 value | 6.36 | 78 | N/A | N/A |
| GC-array transfer; one F32 value | 151.33 | 112 | N/A | N/A |

The GC transfer allocates about 151 KiB of Go memory per fresh command instance,
although the logical plugin buffer has only four bytes. This result includes
instance setup and cleanup. It is not a measurement of a four-byte copy alone.

### First command after module compilation

Device initialization precedes compilation. The first command includes lazy
execution setup and its GPU work. These costs overlap other records and must
not be added to full setup time. Later commands use retained pipelines.

| GPU program | Device init ms | Wasm compile ms | First command ms | Repeated command ms |
| --- | ---: | ---: | ---: | ---: |
| Start section; four F32 values | 56.090 | 0.456 | 10.090 | 0.086 |
| Two buffer kernels; four F32 values | 61.650 | 0.994 | 10.220 | 0.214 |
| TinyGo guest; four F32 values | 53.880 | 0.665 | 10.840 | 0.171 |
| F16 increment; one value | 56.620 | 0.753 | 9.710 | 0.174 |

## WASI Mandelbrot, 64 iterations

All five configured sizes were measured. The default image is 1920×1080.
The larger test is 2560×1440, or 3,686,400 pixels. Each value is the median of
three samples, with three timed images per sample. WASI stdout uses `io.Discard`.
File-system writes and Go process launch are excluded.

The direct CPU program stops updating escaped pixels. The buffer programs
update all pixels in each pass. Their CPU escape check reads both orbit buffers
after each GPU pass. Thus this is a complete program comparison with different
work counts, not a claim about equivalent kernel throughput.

| Image | Retained direct CPU ms | Retained buffer CPU ms | Retained GPU ms | Full setup CPU ms | Full setup buffer CPU ms | Full setup GPU ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 96x64 | 0.266 | 1.240 | 8.926 | 1.250 | 2.778 | 93.128 |
| 256x192 | 1.939 | 8.805 | 17.781 | 2.843 | 9.545 | 109.521 |
| 512x512 | 10.380 | 46.169 | 57.663 | 10.876 | 47.601 | 147.009 |
| 1920x1080 | 77.947 | 364.305 | 387.272 | 76.535 | 372.970 | 483.288 |
| 2560x1440 | 132.788 | 811.829 | 702.237 | 135.753 | 795.841 | 823.653 |

The direct CPU was faster at every tested size. At 1440p the GPU was faster
than buffer CPU fallback. It was still slower than the direct CPU program.

CPU fallback images matched direct CPU images exactly. GPU differences below
are pixel counts from a complete image comparison outside the timer. Relaxed
float arithmetic changes escape decisions near the set boundary.

| Image | GPU pixels different | GPU Go MiB/image | GPU allocations/image | GPU tracked buffer peak MiB | Readback MiB/image | Seed copies MiB/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 96x64 | 0 | 0.347 | 4104 | 0.26 | 3.0 | 3.0 |
| 256x192 | 4 | 1.003 | 4106 | 2.06 | 24.0 | 24.0 |
| 512x512 | 20 | 4.253 | 4111 | 11.00 | 128.0 | 128.0 |
| 1920x1080 | 198 | 31.909 | 4109 | 87.01 | 1012.0 | 1012.0 |
| 2560x1440 | 366 | 56.503 | 4109 | 154.70 | 1800.0 | 1800.0 |

Transfer figures are rounded by Go's benchmark output. At 1440p the GPU reads
back 1,800 MiB and seeds 1,800 MiB of temporary output per image. Current commit
rules preserve each old output until its pass succeeds. Read-before-write
kernels require those seeds. A future GPU loop with a GPU escape check could
reduce these transfers, but it requires more compiler support.

The retained direct CPU program allocates about 31 KiB of Go memory and 429
objects per HD/1440p image. Buffer fallback allocates 31.76 MiB at HD and
56.35 MiB at 1440p, with 923 objects per image. This is Go allocation cost,
not the total guest or driver memory footprint.

### Mandelbrot first-run costs

| GPU image | Device init ms | Wasm compile ms | WGSL translation ms | Pipeline creation ms | First image ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 96x64 | 59.310 | 2.157 | 0.028 | 0.343 | 17.730 |
| 256x192 | 60.150 | 1.809 | 0.020 | 0.381 | 26.300 |
| 512x512 | 67.880 | 2.166 | 0.028 | 0.439 | 70.430 |
| 1920x1080 | 57.690 | 1.602 | 0.020 | 0.284 | 402.100 |
| 2560x1440 | 59.750 | 1.903 | 0.025 | 0.387 | 716.400 |

## Command demos at larger array sizes

Both commands verify GPU results against native Wago CPU execution. Each table
uses nine repeated calls after a separate first call. Modules and instances
stay loaded. These timings do not include full runtime setup. GPU totals include
transfers and guest copies. They are not comparable to the fresh-instance small
example timings above.

### `cmd/demo`: scalar import

The CPU loop calls the original scalar Wasm kernel for every element. The GPU
uses one shader storage buffer and a readback buffer. These are different loops
and storage arrangements from the buffer API benchmark below.

The heavy kernel has 32 `v = v*0.9990234375 + 0.0009765625` steps. Its largest
observed absolute CPU/GPU difference was 0.00001335144 and passed the demo's
floating-point check. Twice and square results matched exactly.

| Kernel | Elements | CPU ms | GPU total ms | First GPU call ms |
| --- | ---: | ---: | ---: | ---: |
| twice | 1,024 | 0.002 | 0.060 | 10.337 |
| twice | 16,384 | 0.038 | 0.074 | 11.233 |
| twice | 1,000,000 | 1.331 | 0.985 | 10.786 |
| twice | 10,000,000 | 13.470 | 9.252 | 22.633 |
| square | 1,024 | 0.005 | 0.055 | 9.706 |
| square | 16,384 | 0.021 | 0.071 | 9.346 |
| square | 1,000,000 | 1.311 | 1.094 | 10.939 |
| square | 10,000,000 | 13.277 | 9.098 | 22.426 |
| heavy | 1,024 | 0.021 | 0.057 | 10.615 |
| heavy | 16,384 | 0.315 | 0.084 | 9.456 |
| heavy | 1,000,000 | 19.396 | 1.256 | 14.130 |
| heavy | 10,000,000 | 195.653 | 9.331 | 25.369 |

### `cmd/buffer-bench`: buffer API

The direct CPU loop performs math in Wasm memory. GPU samples include a fresh
input set, upload, dispatch, readback, final guest copy, and diagnostic checks.
Stage profiling is enabled, with additional queue waits. This setting differs
from the scalar import and Mandelbrot tables. The polynomial kernel performs
32 `v = v*0.875 + 0.25` steps. All four shaders are compiled from actual Wasm.

| Kernel | Elements | Direct CPU ms | GPU total ms | First GPU call ms |
| --- | ---: | ---: | ---: | ---: |
| twice | 1,024 | 0.001 | 0.097 | 13.239 |
| twice | 16,384 | 0.010 | 0.129 | 14.051 |
| twice | 1,000,000 | 0.769 | 2.031 | 20.984 |
| twice | 10,000,000 | 6.144 | 13.092 | 31.459 |
| square | 1,024 | 0.001 | 0.099 | 14.815 |
| square | 16,384 | 0.014 | 0.168 | 10.860 |
| square | 1,000,000 | 0.772 | 1.556 | 12.647 |
| square | 10,000,000 | 7.820 | 13.476 | 34.695 |
| copy | 1,024 | 0.001 | 0.096 | 11.604 |
| copy | 16,384 | 0.009 | 0.122 | 10.771 |
| copy | 1,000,000 | 0.526 | 1.558 | 12.051 |
| copy | 10,000,000 | 5.456 | 12.981 | 31.606 |
| polynomial | 1,024 | 0.021 | 0.093 | 9.621 |
| polynomial | 16,384 | 0.594 | 0.121 | 9.689 |
| polynomial | 1,000,000 | 18.791 | 1.436 | 10.903 |
| polynomial | 10,000,000 | 187.476 | 14.153 | 32.237 |

At ten million elements, polynomial GPU execution is about 13.2 times faster
than direct CPU. Copy, twice, and square are slower on the GPU in this buffer
API test. The scalar import has a slower CPU reference loop, and therefore
shows different CPU/GPU ratios. Do not combine the two baselines.

Stage details for ten million elements:

| Kernel | Upload ms | Compute/wait ms | Readback ms | GPU total ms | Go allocations | Go bytes | Tracked buffer peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| twice | 4.135 | 0.656 | 3.142 | 13.092 | 66 | 4000 | 228.88 |
| square | 4.190 | 0.658 | 3.060 | 13.476 | 66 | 4000 | 228.88 |
| copy | 4.083 | 0.654 | 3.023 | 12.981 | 67 | 4048 | 228.88 |
| polynomial | 4.196 | 0.665 | 3.208 | 14.153 | 66 | 4000 | 228.88 |

These full-range output kernels have no old-output read, so the optimized path
needs no seed copy. Stage times use host clocks and include waits. GPU device
timestamps are unavailable. The stages do not sum to total time: total includes
other guest and host work, and each column is a separate median.

The raw JSON also records memory, first-call costs, resident dispatches, and a
one-element dispatch into each larger output. Matrix multiplication, SAXPY,
and dot-product reference results remain in [the common benchmark report](../COMMON_BENCHMARKS.md).
Those GPU references use authored WGSL and are not additional example compiler support.

## Checks and raw evidence

- CPU-only tests: passed, with `CGO_ENABLED=0`; no GPU hardware needed.
- Real hardware race tests: passed, with `WAGO_GPU_TEST=1` and `-tags webgpu`.
- All 54 example benchmark cases completed three samples each.
- Both CLI demos completed all 28 hardware cases; each has nine repeated calls.
- `go vet -tags webgpu ./...` and all 11 specification checks passed.
- Image comparisons and path checks distinguish hardware execution from fallback.

Raw records (trailing line spaces removed from text logs):

- [Small example setup](../results/examples-2026-10-08-small-setup.txt)
- [Small example repeats](../results/examples-2026-10-08-small-reuse.txt)
- [Storage setup](../results/examples-2026-10-08-storage-setup.txt)
- [Storage repeats](../results/examples-2026-10-08-storage-reuse.txt)
- [Mandelbrot, 64 iterations](../results/examples-2026-10-08-mandelbrot64.txt)
- [Scalar import demo JSON](../results/examples-2026-10-08-demo.json)
- [Buffer API JSON](../results/examples-2026-10-08-buffer-bench.jsonl)
- [CPU-only benchmark check](../results/examples-cpu-benchmark-check.txt)
- [CPU tests](../results/examples-2026-10-08-cpu-tests.txt)
- [Hardware race tests](../results/examples-2026-10-08-gpu-race-tests.txt)

## Run again

Build the native library first with `./native/build.sh`. Then run these commands
from the repository root. The temporary directory must exist. GPU cases fail
if a required dispatch falls back. The small example benchmarks skip their GPU
cases unless `WAGO_GPU_TEST=1` is set. CPU-only verification is a separate run.

```sh
mkdir -p .cache/tmp
export TMPDIR="$PWD/.cache/tmp" GOMAXPROCS=1 WAGO_GPU_TEST=1

go test -p 1 -tags webgpu ./examples/buffers ./examples/start ./examples/tinygo \
  -run '^$' -bench '^BenchmarkExample$/.*/^SetupAndExecute$' -benchtime=3x -count=3
go test -p 1 -tags webgpu ./examples/storage \
  -run '^$' -bench '^BenchmarkExample$/.*/.*/^SetupAndExecute$' -benchtime=3x -count=3
go test -p 1 -tags webgpu ./examples/buffers ./examples/start ./examples/tinygo \
  -run '^$' -bench '^BenchmarkExample$/.*/^ReuseModule$' -benchtime=100x -count=3
go test -p 1 -tags webgpu ./examples/storage \
  -run '^$' -bench '^BenchmarkExample$/.*/.*/^ReuseModule$' -benchtime=100x -count=3
go test -tags webgpu ./examples/mandelbrot \
  -run '^$' -bench '^BenchmarkMandelbrot' -benchtime=3x -count=3 \
  -args -mandelbrot-iterations=64

go run -tags webgpu ./cmd/demo -bench -reps=9 -require-gpu -json
go run -tags webgpu ./cmd/buffer-bench -kernels=twice,square,copy,polynomial -repeats=9

CGO_ENABLED=0 WAGO_GPU_TEST=0 go test -run '^$' -bench '^BenchmarkExample$' \
  -benchtime=1x ./examples/buffers ./examples/start ./examples/tinygo ./examples/storage
CGO_ENABLED=0 go test -count=1 ./...
WAGO_GPU_TEST=1 go test -race -count=1 -tags webgpu ./...
go vet -tags webgpu ./...
python3 spec/check_spec.py --tinygo
```
