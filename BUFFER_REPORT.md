# Buffer plugin implementation report

Measurements recorded on 2026-10-07. Repository checks updated on 2026-10-08. This is an experimental implementation of the reviewed
`wago_gpu_v1` specification. The ABI remains provisional. The repository is `jtenner/wago-gpu`. `v0.0.0` is the experimental release version.

## Result

Real Wasm functions for `2*x` and `x*x+1` were translated into WGSL, executed on
a hardware GPU, and compared bit for bit with native Wago CPU results. All
required array sizes passed. The fresh-input GPU path was slower than direct
CPU execution at every required size. No end-to-end crossover was found.

The buffer start example, the F16 storage example, and the complete TinyGo guest
run on both paths. The GC and memory64 transfer examples run on the CPU.
The start example also passed both mixed CPU/GPU orders. Wago is an unchanged,
pinned dependency. GPU code, ownership, compiler logic, and cleanup are in this
project and its local native dependency patch.

The shorter example hosts now share one setup and cleanup helper. Two new WASI
Mandelbrot guests provide direct CPU rendering and buffer rendering with explicit
GPU passes and CPU fallback. At the earlier 96×64 pixels and 32 iterations, the
direct Wago CPU, buffer CPU, real GPU, and Wasmtime CPU programs produced identical
PGM files. The GPU run completed all 32 passes with no fallback on the same
NVIDIA RTX 4060 Laptop/Vulkan device listed below. This is a correctness check,
not a Mandelbrot speed measurement. The buffer program reads back both orbit
buffers after each pass; float results near the set boundary can differ on other
grids or devices.

The [Mandelbrot benchmark report](examples/mandelbrot/BENCHMARKS.md) records
separate setup and repeated-command measurements. These results do not change
the earlier array-kernel benchmark tables below.

The [earlier large-image performance report](examples/mandelbrot/PERFORMANCE.md)
records the full-HD/1440p runs and scratch reuse changes. The
[current performance report](examples/mandelbrot/PERFORMANCE_FOLLOWUP.md) records
CPU input reuse, proven GPU copy omission, and unchanged-parameter reuse. The array benchmark
numbers below remain the earlier recorded measurements.

## Implemented behavior

- All 34 canonical imports, all 11 CPU element types, and eight GPU element types.
- Checked memory32, memory64, and Wasm GC-array transfers. Guest views remain inside their checked storage callback and are
  never captured by native callbacks. Immutable GC-array copies are budgeted first.
- Bounded Wasm decoding and straight-line lowering, immutable stack values,
  saved-index tracking, public typed errors, and actual read/write sets.
- Source-verified contracts, separate bounded pending metadata and WGSL storage,
  module/instance pipeline ownership, and a controlled host without adoption or
  import overrides.
- Explicit fallback with current CPU data, private GPU outputs, one commit
  decision, device quarantine, lost-content detection, and full-set repair.
- Runtime and instance byte limits, peak counters, scratch reuse, pipeline reuse,
  and retained-resource charging after uncertain completion.
- Exact integer transport and exact float copies. F16 scalar conversion uses
  integer operations. Float arithmetic requires per-kernel relaxed consent.
- Diagnostics that preserve the last dispatch through CPU scalar loops and
  distinguish guest copies, device transfers, seed copies, and submission.

## Tests and hardware evidence

CPU tests run without a native library or a GPU. Generated Wasm edge-case tests
need `wasm-tools`; these tests skip with a stated reason when the tool is absent.
It was present for this run.

| Check | Evidence |
| --- | --- |
| CPU-only build and tests | [CPU log](results/buffer-tests-cpu.txt), `CGO_ENABLED=0 go test -count=1 ./...` |
| CPU/fake/native tests with race detector | [GPU race log](results/buffer-tests-hardware-race.txt), `WAGO_GPU_TEST=1 go test -count=1 -race -tags webgpu ./...` |
| Pinned callback checks | [cgo check log](results/binding-cgocheck.txt), `GOEXPERIMENT=cgocheck2 WAGO_GPU_TEST=1 go test -count=1 -tags webgpu -run TestBinding .` |
| Full WAT and TinyGo demonstrations | [demo log](results/buffer-demos.txt) |
| Bounded compiler fuzz test | [fuzz log](results/buffer-fuzz.txt), 127,278 executions in the recorded 10-second run |
| ABI, WAT, compiled guest, and document checks | [validation report](spec/validation.json), `python3 spec/check_spec.py --tinygo` |
| Required benchmark cases | [raw JSONL](results/buffer-bench.jsonl), eight cases, full-array result checks after every GPU sample |
| Simplified hosts and WASI guests, CPU-only | [example CPU log](results/examples-cpu-tests.txt), full `CGO_ENABLED=0` suite |
| Simplified hosts and WASI guests, CPU race tests | [example CPU race log](results/examples-cpu-race.txt), full suite |
| Simplified hosts and WASI guests, real GPU race tests | [example hardware log](results/examples-hardware-race.txt), full suite with `WAGO_GPU_TEST=1` |
| Default WASI Mandelbrot images | [commands and hashes](results/mandelbrot-example-checks.txt), CPU, fallback, real GPU, and Wasmtime |

The tests include invalid ranges, missing/disabled devices, unsupported code,
public error wrapping, metadata eviction, durable invalid contracts, failed
instantiation, repeated execution, concurrent instances, resource limits,
unused declared aliases, actual-write versions, multiple-output failure,
cancellation before commit, outer Wago cancellation after commit, and shutdown
while a fake dispatch is blocked.

Real hardware tests include all four start-path combinations, an in-place
65-element update with reused scratch storage, integer constant/index stores
through all six integer GPU types, exact F32 copies after a separate arithmetic
use, and all 65,536 F16 patterns for widening and conversion-only round trips.
The F16 tests compare bits, with the specified NaN canonicalization.

Native failure tests reject invalid encoding, invalid mapped access, and a
resource destroyed between encoding and submission. A later successful queue
notification does not erase an earlier error. Pending map callbacks survive GC
and allocation pressure, both on completion and on buffer destruction. These
are actual native calls, separate from fake-backend transaction tests.

## Dependency repairs

The original binding could not satisfy the callback and error-reporting gates.
The GPU build uses a local copy of `oliverbestmann/webgpu` v1.36.0 with explicit
resource release, pinned callback contexts, completed validation/allocation
error scopes, checked submission, and device-loss notification.

The native library is pinned to wgpu-native
`d2e3330ade4ae1bb238d76b485926f067e7ee64c` (v29.0.0.0).
Its local Rust patch sends submission, poll, and mapped-range errors to the
error sink instead of aborting on those paths. The pinned sink supports
validation, allocation, and device-loss reporting. It rejects an Internal
error-scope filter; the implementation does not request that filter.

See [patch notes](third_party/webgpu/PATCHES.md),
[the native patch](native/wgpu-native-errors.patch), and
[the reproducible build script](native/build.sh). Native binaries and build
caches are ignored by Git. Wago has no local patch.

Earlier development logs include rejected backend attempts. They are not
acceptance evidence; the final logs listed above are the acceptance runs.

## Benchmark settings

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB reported memory.
- Driver: NVIDIA 550.163.01. Backend: Vulkan, discrete hardware adapter.
- CPU: AMD Ryzen 7 8845HS, 8 cores / 16 logical CPUs.
- OS: Debian GNU/Linux 13, Linux 6.12.111+deb13-amd64, x86-64.
- Go: 1.27.1. Wago: `v0.1.0-beta.12.0.20261007220511-7aa401f29a33`.
- One benchmark process; one host, module, and instance per kernel/size case.
- Workgroup size: 256. One invocation per element. Both kernels use F32.
- One first GPU sample and three repeated fresh-input GPU samples per case.
  Tables show medians for repeated samples. Direct CPU has a first sample and
  three repeated samples. Buffer-import CPU has one sample.
- `ProfileStages=true`. Stage separation adds queue waits. GPU device timestamps
  are unavailable. The compute column is host encoding/submission/wait time,
  **not** isolated device execution time. No device-time result is claimed.
- No clock pinning, thermal control, or statistical confidence interval. These
  are local experiment results. Small timings and single resident samples have
  substantial measurement noise.

Fresh-input end-to-end time includes guest-to-plugin set, upload, full output
seed copy, dispatch, readback, and plugin-to-guest copy. It excludes host/module
creation, guest fixture initialization, and result verification. Those first
costs are reported separately. Direct CPU writes ordinary guest memory and
avoids per-element host imports. Buffer CPU calls the same typed imports as the
fallback runner. It is an additional baseline; its high cost must not replace
the direct CPU baseline in speedup claims.

## CPU and GPU elapsed time

All times below are milliseconds.

| Kernel | Elements | CPU first | CPU repeated median | Buffer CPU | GPU first | GPU repeated median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| twice | 1,024 | 0.0027 | 0.0015 | 1.3165 | 13.7171 | 0.2747 |
| twice | 16,384 | 0.0104 | 0.0117 | 21.0359 | 14.9089 | 0.3064 |
| twice | 1,000,000 | 1.1117 | 0.5920 | 1300.2613 | 20.4350 | 1.6727 |
| twice | 10,000,000 | 17.8916 | 6.1956 | 12897.2055 | 159.2863 | 89.9204 |
| square | 1,024 | 0.0025 | 0.0027 | 2.1170 | 14.4092 | 0.3094 |
| square | 16,384 | 0.0136 | 0.0143 | 34.1386 | 13.8430 | 0.1843 |
| square | 1,000,000 | 0.9984 | 0.7719 | 1926.4476 | 21.2538 | 1.5393 |
| square | 10,000,000 | 20.0111 | 8.0027 | 19840.8313 | 155.6506 | 90.2204 |

The first GPU sample includes first buffer allocation and first driver execution
costs. Device creation, Wasm compilation, WGSL translation, and pipeline
creation occur before that sample. The CPU fallback loop no longer allocates
per element for non-cancellable calls; host-call and validation costs still
make it much slower than the direct-memory CPU loop.

## Repeated GPU stages

Each column is its own median over three runs; these medians need not sum to the
median total. Guest copies, conversion, and host work also contribute to total.

| Kernel | Elements | CPU→GPU | Device seed copy | Compute host stage | GPU→CPU | End-to-end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| twice | 1,024 | 0.0463 | 0.0527 | 0.0811 | 0.0550 | 0.2747 |
| twice | 16,384 | 0.0470 | 0.0401 | 0.0694 | 0.0627 | 0.3064 |
| twice | 1,000,000 | 0.4996 | 0.1049 | 0.1047 | 0.3650 | 1.6727 |
| twice | 10,000,000 | 38.7907 | 10.8355 | 10.8546 | 23.6124 | 89.9204 |
| square | 1,024 | 0.0662 | 0.0421 | 0.0683 | 0.0455 | 0.3094 |
| square | 16,384 | 0.0412 | 0.0356 | 0.0389 | 0.0353 | 0.1843 |
| square | 1,000,000 | 0.4378 | 0.0735 | 0.0742 | 0.3715 | 1.5393 |
| square | 10,000,000 | 38.8788 | 10.8282 | 10.8463 | 23.6469 | 90.2204 |

## Resident dispatch and small prefix

These single dispatch samples exclude all host/device transfers and guest
copies from their timing. A readback and full-array check follow outside the
measurement. The prefix case updates one element and must preserve the rest of
the full-sized output. This implementation always seeds an actual output.

| Kernel | Buffer elements | Resident full dispatch ms | One-element dispatch ms | Device seed bytes in prefix |
| --- | ---: | ---: | ---: | ---: |
| twice | 1,024 | 0.1097 | 0.1284 | 4,096 |
| twice | 16,384 | 0.1585 | 0.1018 | 65,536 |
| twice | 1,000,000 | 0.2440 | 0.1799 | 4,000,000 |
| twice | 10,000,000 | 22.5126 | 9.5945 | 40,000,000 |
| square | 1,024 | 0.1299 | 0.1152 | 4,096 |
| square | 16,384 | 0.0817 | 0.1036 | 65,536 |
| square | 1,000,000 | 0.2179 | 0.2502 | 4,000,000 |
| square | 10,000,000 | 20.3019 | 9.2317 | 40,000,000 |

A resident one-million-element dispatch was faster than the measured direct CPU
loop. This is not an end-to-end speedup: it requires data already on the GPU and
omits result readback. There is no established general minimum profitable size.
The 10-million-element resident and small-prefix results show the cost of full
output seeding. The complete two-kernel demo also verifies GPU-resident
intermediate data; it is not a timing benchmark.

## Compilation and allocation

Compilation total includes native Wago compilation, the source hook, and GPU
pipeline setup. Translation and pipeline timings are contained in that total;
they are not additional costs. Native Wasm compilation alone is not isolated.
Each case has a fresh host. Wago prepared-artifact cache eligibility is disabled
by the source-transform hook. Device creation is outside compilation total.

| Kernel | Elements | Device init ms | Compile total ms | WGSL translation ms | Pipeline ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| twice | 1,024 | 96.6705 | 0.6185 | 0.0141 | 0.2208 |
| twice | 16,384 | 57.5270 | 0.5364 | 0.0131 | 0.2057 |
| twice | 1,000,000 | 62.3478 | 0.6072 | 0.0147 | 0.2474 |
| twice | 10,000,000 | 59.0940 | 0.5430 | 0.0151 | 0.2188 |
| square | 1,024 | 70.9919 | 0.5055 | 0.0132 | 0.1877 |
| square | 16,384 | 57.7616 | 0.4989 | 0.0139 | 0.1845 |
| square | 1,000,000 | 56.6902 | 0.5595 | 0.0151 | 0.2396 |
| square | 10,000,000 | 62.7859 | 0.5181 | 0.0150 | 0.1925 |

Go allocation counts below cover one repeated fresh-input GPU operation. Counts
and bytes are medians of the three samples. Go memory statistics are
process-wide; background runtime work can affect them. Native/driver allocations
are outside Go heap statistics. Raw results also contain CPU allocation counts,
first-run allocation counts, HeapAlloc, and HeapSys.

| Kernel | Elements | GPU Go allocations | GPU Go allocated bytes | Retained tracked bytes | Peak tracked bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| twice | 1,024 | 72 | 3848 | 20,480 | 24,576 |
| twice | 16,384 | 72 | 3848 | 327,680 | 393,216 |
| twice | 1,000,000 | 89 | 7192 | 20,000,000 | 24,000,000 |
| twice | 10,000,000 | 86 | 6688 | 200,000,000 | 240,000,000 |
| square | 1,024 | 72 | 3848 | 20,480 | 24,576 |
| square | 16,384 | 72 | 3832 | 327,680 | 393,216 |
| square | 1,000,000 | 72 | 3848 | 20,000,000 | 24,000,000 |
| square | 10,000,000 | 87 | 6736 | 200,000,000 | 240,000,000 |

The largest tracked reservation peak was 240,000,000 bytes (228.9 MiB), within
the default 256 MiB instance cap. The process maximum RSS was
501,876 KiB (490.1 MiB). The full benchmark process took
38.16 seconds. RSS includes guest memory, native compiler/driver state,
and allocations outside the plugin budget. See [process measurements](results/buffer-bench-process.txt).

## Limits and acceptance scope

The implementation is useful for this experiment. It is not a claim that every
release-matrix failure can be reproduced on this one GPU.

- GPU builds are Linux-only. Linux/amd64 on the named NVIDIA GPU was tested.
  Other platforms use CPU fallback. No other driver/vendor was tested.
- Actual device removal, a driver hang, native allocator exhaustion, and native
  callbacks arriving after runtime shutdown were not induced on hardware.
  Device loss, cancellation/commit boundaries, allocation limits, and shutdown
  ordering have controlled tests. Native pending-map destruction and GC have
  hardware tests. These are different evidence classes.
- The native dependency has local safety patches. They need independent review
  and upstream work before a public release. Other unused binding operations
  are outside this plugin's tested subset.
- GPU timestamps and isolated device execution time are unavailable. A Go
  context cannot forcibly stop a native driver call.
- General control flow, integer arithmetic, cross-element access, reductions,
  matrices, automatic replacement, and GPU 64-bit scalar types are outside v1.
- Float arithmetic follows the documented relaxed contract. Exact copy tests do
  not establish strict Wasm float arithmetic on GPU.
- The decoder rejects modules beyond its supported form or limits. Such modules
  can lack verified v1 metadata and return INVALID_KERNEL. Unsupported kernel
  instructions in a verified module can use CPU fallback.
- TinyGo's complete F32 management path passed. It is not a claim that all guest
  languages or GC-aware source compilers can emit accepted kernels.
- Scratch is reused, but upload/readback resources are still allocated per
  transfer. Full-output seeding is conservative. The global device lock
  serializes plugin work; concurrent instances are safe but do not dispatch
  concurrently on this backend.

## Wago API changes

None were made. The controlled host uses public compilation, lifecycle,
caller-identity, cancellation, and checked guest-storage APIs. It exposes only
source compile routes and no import overrides. These host restrictions are
required by the pinned Wago API.

Generic final-module export/memory metadata and source-bound artifact adoption
would permit broader host integration. A checked GC-array range-copy API would
avoid the hidden full immutable-array copy. These are proposed generic
improvements, not GPU code in Wago and not requirements for this restricted
experiment.

## Next three improvements

1. Prove full-output overwrite and absence of old-value reads so eligible
   kernels can omit seed copies. Measure it with resident chains and prefix work.
2. Reuse upload/readback storage with the same budget and callback lifetime
   rules. Add device timestamp queries so compute and transfer costs can be
   separated from host waits.
3. Review and upstream the native safety patches, then test a second GPU vendor
   and expand the timeout, loss, and shutdown fault matrix before ABI freeze.
