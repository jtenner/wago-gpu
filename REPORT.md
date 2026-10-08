# Initial experiment report: wago-gpu

This report records the first implementation. See [IMPROVEMENTS.md](IMPROVEMENTS.md) for the later batch, timeout, heavy-kernel, and selection work. Numbers below were not replaced by later measurements.

The primary test passed. Two real Wasm functions were compiled into WGSL, executed on a physical GPU, and checked against Wago's native CPU results. Wago was not modified.

Small arrays were slower on the GPU. At one million and ten million elements, repeated GPU calls were faster in this run. First GPU calls were slower at every required size.

## Test system and method

- Date: 2026-10-07. The main measurement started at 23:14:42 UTC.
- GPU used: NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB reported memory, 55 W power limit.
- Driver: NVIDIA 550.163.01. GPU backend: Vulkan.
- CPU: AMD Ryzen 7 8845HS. The AMD integrated GPU was present but was not tested.
- OS: Debian 13.7, Linux 6.12.111+deb13-amd64.
- Go: 1.27.1, linux/amd64. `GOMAXPROCS=1`; default Go GC settings.
- Wago: `7aa401f29a33` (`v0.1.0-beta.12.0.20261007220511-7aa401f29a33`).
- WebGPU binding: `github.com/cogentcore/webgpu` v0.23.0.
- Workgroup size: 256. Each invocation processes one element.
- Each case used a fresh Wago runtime, GPU device, and pipeline. Driver caches were not cleared.
- Each case measured one first call, then 50 repeated calls per path. CPU batches ran before GPU batches.
- Clocks and CPU affinity were not locked. The CPU governor reported `performance`. Other system activity was not disabled.
- Setup, Go GC, result verification, and memory-stat reads were outside call timing. Race instrumentation was disabled during benchmarks.
- Every output was compared before and after repeated execution. Inputs were bounded binary fractions, with exact results for both functions.

See [environment.txt](results/environment.txt) for the hardware and driver output. All figures below came from saved runs.

## Native CPU versus GPU

Times are milliseconds. Repeated values are medians of 50 calls. GPU time includes the guest import, upload, command submission, compute, download, and guest result commit. CPU time is one Wago invocation of the original Wasm array loop. It is not a Go arithmetic loop.

| Function | Elements | First CPU | Repeated CPU | First GPU | Repeated GPU total | CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| twice | 1,024 | 0.1004 | 0.0015 | 5.3759 | 0.0607 | 0.03x |
| twice | 16,384 | 0.0225 | 0.0210 | 5.1969 | 0.0800 | 0.26x |
| twice | 1,000,000 | 1.4609 | 1.3209 | 7.2959 | 1.0358 | 1.28x |
| twice | 10,000,000 | 27.3682 | 15.2250 | 54.9139 | 11.1536 | 1.37x |
| square | 1,024 | 0.0142 | 0.0015 | 5.3197 | 0.0570 | 0.03x |
| square | 16,384 | 0.0463 | 0.0212 | 5.8120 | 0.0972 | 0.22x |
| square | 1,000,000 | 2.7693 | 1.5384 | 8.6318 | 1.0547 | 1.46x |
| square | 10,000,000 | 27.1288 | 15.2192 | 56.0204 | 11.4656 | 1.33x |

`twice` means `2*x`; `square` means `x*x+1`. A ratio above 1 means that GPU execution was faster.

These are local measurements, not universal speed claims. A separate Go benchmark run measured 12.91 ms CPU and 11.27 ms GPU for ten million `twice` elements. The main run measured 15.22 ms CPU and 11.15 ms GPU. This variation matters near the crossover point.

The first-call columns include lazy execution and initial buffer costs. Device creation and compilation are separate below. They are not cold machine or cold driver-cache measurements.

Raw data: [benchmark.json](results/benchmark.json). Independent Go benchmark output: [go-benchmarks.txt](results/go-benchmarks.txt).

## Transfer and compute stages

These are mean host wall times, in milliseconds, from a separate 50-call profile run. Extra waits separate the stages. The compute column includes command encoding, submission, and waiting for completion. It is not a hardware timestamp for the shader alone.

| Function | Elements | CPU to GPU | Compute stage | GPU to CPU | Commit to guest | Profiled total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| twice | 1,024 | 0.0444 | 0.0405 | 0.0337 | 0.0004 | 0.1199 |
| twice | 16,384 | 0.0692 | 0.0599 | 0.0543 | 0.0031 | 0.1877 |
| twice | 1,000,000 | 0.5517 | 0.0869 | 0.3584 | 0.1704 | 1.1720 |
| twice | 10,000,000 | 6.0858 | 0.4831 | 3.0936 | 2.2897 | 11.9681 |
| square | 1,024 | 0.0390 | 0.0446 | 0.0370 | 0.0006 | 0.1224 |
| square | 16,384 | 0.0527 | 0.0455 | 0.0434 | 0.0029 | 0.1456 |
| square | 1,000,000 | 0.5271 | 0.0737 | 0.3557 | 0.1992 | 1.1586 |
| square | 10,000,000 | 6.1737 | 0.4681 | 3.1028 | 2.2290 | 11.9889 |

Transfers and the final host copy dominate at ten million elements. The compute stage takes about 0.47–0.48 ms. Upload, download, and guest commit take about 11.5 ms combined.

The normal path combines compute and result-copy commands. Do not compare the sum of profiled stage means directly with a normal-call median. Raw data: [stages.json](results/stages.json).

## Compilation and startup

Each row is one observation, in milliseconds. Native Wasm compilation used public `wago.Compile` without the GPU plugin. Plugin compilation used `Runtime.Compile`, including source inspection, Wago compilation, and pipeline creation. The WGSL and pipeline columns are components of plugin compilation, not additional costs.

| Function | Elements | Native Wasm compile | Wasm to WGSL | WGSL and pipeline | Plugin compile total | Device creation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| twice | 1,024 | 0.2887 | 0.0107 | 0.5790 | 0.8157 | 158.6038 |
| twice | 16,384 | 0.2720 | 0.0107 | 0.3753 | 0.6190 | 112.8568 |
| twice | 1,000,000 | 0.2378 | 0.0103 | 0.2988 | 0.5266 | 104.1291 |
| twice | 10,000,000 | 0.2247 | 0.0104 | 0.3023 | 0.5552 | 118.5696 |
| square | 1,024 | 0.2359 | 0.0102 | 0.3164 | 0.5385 | 110.4636 |
| square | 16,384 | 0.2369 | 0.0112 | 0.3373 | 0.5740 | 114.0550 |
| square | 1,000,000 | 0.2313 | 0.0130 | 0.3586 | 0.6232 | 134.8238 |
| square | 10,000,000 | 0.2350 | 0.0106 | 0.3037 | 0.5371 | 119.0829 |

Device creation took about 104–159 ms. Native Wasm compilation took about 0.22–0.29 ms. These small functions took about 0.01 ms to translate. Native driver caches may reduce pipeline creation costs in later cases.

## Allocations and memory

The Go benchmark measured zero Go allocations per repeated native CPU call. Repeated GPU calls used about 968–1,019 bytes and 39–44 Go allocations. Allocation size did not grow with the array size. Most remaining small allocations come from host-call and native binding work.

The table uses the main GPU run and a separate CPU-only process. Values are MiB. RSS is a process snapshot after repeated calls, not peak memory. Both runs use the same three guest arrays. The GPU buffer column includes storage and host-visible result buffers; it excludes driver staging and internal allocations.

| Function | Elements | Guest memory | Plugin buffer capacity | CPU-only process RSS | GPU process RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| twice | 1,024 | 0.06 | 0.01 | 13.73 | 163.91 |
| twice | 16,384 | 0.19 | 0.12 | 12.75 | 170.14 |
| twice | 1,000,000 | 11.50 | 7.63 | 23.54 | 183.29 |
| twice | 10,000,000 | 114.50 | 76.29 | 127.36 | 372.59 |
| square | 1,024 | 0.06 | 0.01 | 14.86 | 172.00 |
| square | 16,384 | 0.19 | 0.12 | 14.92 | 172.33 |
| square | 1,000,000 | 11.50 | 7.63 | 25.12 | 183.55 |
| square | 10,000,000 | 114.50 | 76.29 | 127.33 | 374.65 |

The main GPU process had about 0.20–0.21 MiB of live Go heap at these snapshots. This excludes native Wasm mappings and native driver memory. At ten million elements, the guest reserves 114.5 MiB and the plugin has 76.29 MiB of logical buffer capacity. Driver and staging memory increase process use further.

The `CPU` memory fields inside `benchmark.json` include an already loaded GPU device and its first-call buffers. They are not standalone CPU memory measurements. Use [cpu-only.json](results/cpu-only.json) for the separate CPU-only process. Compilation allocation counts and first-call allocation counts are also saved in the JSON files.

## Optimization results

- Pipelines are cached by generated WGSL within a plugin. Equal kernels in live modules share a pipeline.
- GPU storage and result buffers are reused. A larger call replaces them; smaller calls use a binding with the exact active length.
- Checked guest input is passed directly to the upload API. A completed mapped result is copied directly to the checked output view.
- There is no array-sized Go allocation in the GPU execution path. The compiler and array processing have linear cost.
- Upload and download still occur on every call. Omitting them would require an explicit data-lifetime contract. The changed-input test checks that reuse never assumes stale input is valid.

A separate 100-call size scan found these first sampled GPU gains:

| Function | Last slower sampled size | First faster sampled size | CPU median at faster size | GPU median at faster size |
| --- | ---: | ---: | ---: | ---: |
| `2*x` | 131,072 | 262,144 | 0.3409 ms | 0.3094 ms |
| `x*x+1` | 262,144 | 393,216 | 0.5083 ms | 0.4363 ms |

These are sample points, not exact minimum thresholds. Scheduling, clock changes, and transfer behavior can move the boundary. Raw data: [crossover.json](results/crossover.json).

## What works and what was checked

The compiler translates real Wasm instructions, including operand order and exact constant bits. The plugin uses Wago's public registration, source, compile, caller, guest-memory, and close APIs. The guest performs its own CPU fallback. No GPU output is written after a reported device, shader, or result-read failure.

The following checks passed:

- CPU tests with `CGO_ENABLED=0`.
- CPU and hardware tests with Go's race detector.
- Instruction translation, unsupported instructions, wrong signatures, malformed source, and decode limits.
- Invalid input and output ranges, count overflow, partial overlap, and in-place calls.
- Missing-device, shader-failure, and execution-failure injection. These tests use a labeled test backend, not a GPU claim.
- Actual GPU execution of both functions, with 1, 257, 1,024, 16,384, and one million elements in the hardware suite.
- All four required benchmark sizes on the physical NVIDIA GPU, including ten million elements.
- Changed input at a fixed size, buffer growth and shrink use, repeated calls, and concurrent instances.
- Actual native shader rejection for invalid WGSL.
- Pipeline sharing, module close before instance close, live-resource runtime shutdown, and bounded abandoned preparations.
- `go vet` for CPU and WebGPU builds.
- A 15-second compiler fuzz run: 763,201 executions, with no failure.
- CPU fallback demonstration and real GPU demonstration.

Logs: [CPU tests](results/cpu-tests.txt), [CPU race tests](results/race-tests.txt), [GPU race tests](results/hardware-tests.txt), and [fuzz run](results/fuzz.txt).

## Limits and Wago API changes

This is not a full Wasm compiler. It supports one selected `(f32) -> f32` function per module and only the stated instruction subset. It has no control flow, extra locals, function calls, automatic replacement, or general optimizer. It does not support shared/imported memory, Memory64, or multiple memories on the GPU path.

Strict Wasm floating-point equivalence is not established. GPU use requires `RelaxedFloat: true`. The checked examples use exact inputs. Non-finite values, subnormals, fused operations, and reassociation can differ. The README links the applicable WGSL rules.

GPU calls are synchronous and serialized across instances. The current binding provides no usable device-loss callback in this setup. There is no driver-hang timeout, cancellable GPU submission, or device timestamp measurement. Optional descriptor paths with binding defects are avoided. Only the NVIDIA Vulkan device was tested; the AMD device and other operating systems were not tested.

Required Wago changes: **none**. All GPU code stays in this project. The optional API gap is a missing terminal event when `PreparedCompile.Close` abandons work. A generic abandonment event with `CompilationIdentity` would permit immediate cleanup. The current 16-entry pending cache bounds memory and safely falls back for evicted work.

## Next three improvements

1. Add explicit GPU-resident batches. Upload once, run several kernels, and download once. Keep memory ownership and updates explicit.
2. Use a maintained binding with device-loss handling and timestamp queries. Measure the shader itself and add bounded failure handling.
3. Add more work per element and test CPU/GPU selection near the measured boundary. Prefer useful arithmetic workloads before adding a large optimizer.
