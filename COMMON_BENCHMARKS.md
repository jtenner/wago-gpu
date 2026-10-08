# Common workloads and reduced work

Measured on 2026-10-08. GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB,
Vulkan driver 550.163.01. CPU: AMD Ryzen 7 8845HS. OS: Debian 13,
Linux 6.12.111. Go: 1.27.1. Wago remains pinned and unchanged.

## Scope

**The vector copy, twice, square, and 32-step arithmetic kernels are compiled
from actual Wasm by the plugin. Matrix multiplication, SAXPY, and dot product
use authored WGSL reference shaders in a benchmark test.** Their CPU references
are real Wasm functions executed by native Wago. The reference tests use the
same private GPU backend and checked error/completion protocol. They do not
add plugin compiler instructions, guest imports, or a public shader interface.
Matrix loops, arbitrary indexing, and reduction barriers are still outside the
plugin compiler subset. These measurements establish backend capacity only.

## Less work with the same contract

- Retain at most two native binding groups. Keys include the pipeline and all
  live buffer wrappers. Closing a member or pipeline releases its groups first.
  The cache avoids repeated native group creation and buffer-size queries.
  It cannot grow with the number of calls or retain a freed buffer.
- For an output with covering stores and no old-value reads, seed only the
  unchanged tail during a partial dispatch. A full dispatch still skips the
  seed. Read-before-write outputs still copy all old contents. A 512-element
  dispatch into a 513-element F32 output now seeds 4 bytes instead of 2,052.
- CPU-only bulk transfers and fallback with current CPU data check the entry
  deadline without a timer allocation.
  GPU waits retain a cancellable timer. Invalid ranges and valid zero-count
  calls keep their specified order. The final commit check includes lock wait.
- Benchmark CPU runners copy input once, execute a native Wasm loop, and set
  output once. They use two bulk imports instead of several scalar host calls
  per element. Partial output updates retain their tails. Scalar imports remain
  available and tested. This is a guest-runner improvement, not automatic
  replacement of an arbitrary guest's CPU runner.

Private outputs, full error checks, cancellation checks, resource charges,
loss detection, and the single commit decision remain in place. No successful
queue notification alone can publish output. No Wago API change was needed.

## Compiled Wasm vectors

`GOMAXPROCS=1`; one host/module/instance per case; three repeated samples.
GPU end-to-end includes guest-to-plugin set, device upload, dispatch, readback,
and final guest copy. Resident dispatch excludes transfers and final copy.
ProfileStages is enabled, so transfer stages use separate checked submissions.
The direct CPU baseline runs in guest memory. The bulk CPU measurement includes
input copy and output set, but no dispatch/fallback-preparation call. It is a
single sample; CPU repeated and GPU repeated columns use medians.

The arithmetic kernel repeats `v = v * 0.875 + 0.25` 32 times per element.
The generated Wasm contains those instructions. It is not a fixed GPU shader.
Inputs use bounded dyadic F32 values. Copy, twice, and one-pass square compare
exact bits. Arithmetic uses finite-value tolerance
`8e-6 * max(1, abs(CPU))`; relaxed GPU F32 arithmetic can differ.
Every first, repeated, and resident GPU result is checked outside timed work.
These samples precede the final CPU-fallback-only timer change. It does not
change the work in the direct CPU, bulk CPU, or GPU measurements. The
Mandelbrot after-run includes that final change.

| Kernel | Elements | Direct CPU ms | Bulk CPU ms | GPU total ms | Resident GPU ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| twice | 1,024 | 0.0012 | 0.0079 | 0.0988 | 0.0348 |
| twice | 16,384 | 0.0109 | 0.0219 | 0.1163 | 0.0351 |
| twice | 1,000,000 | 0.7641 | 1.2486 | 1.7561 | 0.1358 |
| twice | 10,000,000 | 5.8688 | 13.6876 | 13.0969 | 0.5132 |
| square | 1,024 | 0.0016 | 0.0045 | 0.1289 | 0.0439 |
| square | 16,384 | 0.0145 | 0.0188 | 0.2414 | 0.0683 |
| square | 1,000,000 | 0.7268 | 1.5810 | 1.3496 | 0.0853 |
| square | 10,000,000 | 7.9118 | 16.7511 | 14.8292 | 0.5125 |
| copy | 1,024 | 0.0013 | 0.0045 | 0.1156 | 0.0366 |
| copy | 16,384 | 0.0110 | 0.0152 | 0.1216 | 0.0389 |
| copy | 1,000,000 | 0.5490 | 1.3834 | 1.5841 | 0.1009 |
| copy | 10,000,000 | 5.7393 | 12.9233 | 19.2675 | 0.4965 |
| polynomial | 1,024 | 0.0197 | 0.0239 | 0.1163 | 0.0380 |
| polynomial | 16,384 | 0.3113 | 0.3065 | 0.1540 | 0.0460 |
| polynomial | 1,000,000 | 19.1405 | 20.8675 | 1.5972 | 0.1609 |
| polynomial | 10,000,000 | 185.7637 | 193.0739 | 13.6876 | 0.6377 |

Copy and the small arithmetic kernels still favor direct CPU execution with
fresh inputs. The 32-step arithmetic kernel has enough work to amortize the
transfers. One million elements took 19.14 ms on direct CPU and 1.60 ms on GPU;
ten million took 185.76 ms and 13.69 ms. Those GPU gains use the plugin compiler.

The earlier scalar CPU runner took 10,418.39 ms for ten million twice elements
and 15,357.40 ms for square. The new bulk runner took 13.69 ms and 16.75 ms.
These are recorded single samples from separate runs on the same machine.
Both algorithms are linear; the old runner spent time crossing the host boundary.
See the [earlier raw samples](results/perf-array-bench.jsonl).

### First costs, stages, and memory

For ten million elements (median repeated stages; host clock times):

| Kernel | Device init ms | Wasm + pipeline compile ms | First GPU total ms | Upload ms | Dispatch ms | Download ms | Peak tracked MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| twice | 66.21 | 0.63 | 44.48 | 4.13 | 0.65 | 3.16 | 228.88 |
| square | 57.31 | 0.71 | 45.43 | 4.14 | 0.66 | 3.35 | 228.88 |
| copy | 56.24 | 0.60 | 32.29 | 5.38 | 0.69 | 3.49 | 228.88 |
| polynomial | 62.59 | 1.15 | 43.53 | 4.24 | 0.72 | 3.26 | 228.88 |

Stage medians need not sum to the total median. Compilation includes native
Wago compilation plus shader/pipeline work. The raw JSON separates translation,
pipeline creation, and device initialization. First GPU total includes lazy
buffer allocation; device initialization and compilation precede it.

The ten-million-element bulk CPU sample uses 5 Go allocations and 688 allocated
bytes. A small warmed resident dispatch uses 19 allocations and about 1.4 KiB.
Large calls can include runtime allocation changes. Raw JSON records every
sample's allocation count, allocated bytes, HeapAlloc, and HeapSys. Native
allocations are not included in Go heap counters. Peak plugin buffer storage
remains at most 228.89 MiB in these vector cases, within the 256 MiB limit.

### Observed crossover

A separate arithmetic size test uses nine repeated CPU and GPU samples per
size. The first sampled GPU win is 8,192 elements. This is a coarse observation,
not the exact smallest size or a portable scheduling threshold.

| Elements | Direct CPU µs | GPU total µs |
| --- | ---: | ---: |
| 2,048 | 39.02 | 95.23 |
| 4,096 | 78.50 | 95.80 |
| 8,192 | 160.08 | 107.94 |
| 12,288 | 243.77 | 105.09 |
| 16,384 | 401.18 | 189.51 |
| 32,768 | 621.57 | 133.24 |

## Matrix reference benchmarks

Dense, row-major F32 `C = A * B`. CPU naive uses column-strided B loads.
CPU transposed uses an existing transposed B. The transpose-and-matmul column
also times creation of that transpose on every call. The GPU tile variant uses
16×16 shared-memory tiles and 256 invocations per workgroup. Dimensions are
multiples of 16; the hardware correctness tests also cover odd dimensions in
the naive shader. These are simple loops, not BLAS or vendor GEMM comparisons.
No tensor cores, FP16 inputs, or specialized matrix instructions are used.
Matrix multiplication performs the expected cubic work; no extra pairwise
scan or repeated allocation was added.

Times are medians of three samples, each averaging three executions.
GPU total uploads both inputs and downloads the output on every execution.
Resident GPU time includes checked encoding, submission, and completion,
but excludes upload and download. Setup/compile time is excluded from warm
columns and reported separately as `first-ms` in the raw results. A new device
and pipeline are created for each case. Driver cache state is not controlled.

| Dimension | CPU naive ms | CPU transposed ms | CPU transpose + multiply ms | GPU naive total ms | GPU tiled total ms | GPU tiled resident ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64×64 | 0.2675 | 0.2500 | 0.2603 | 0.1352 | 0.1228 | 0.0366 |
| 128×128 | 2.2808 | 2.0621 | 2.0692 | 0.1927 | 0.1609 | 0.0446 |
| 256×256 | 18.4420 | 16.3872 | 16.4719 | 0.2958 | 0.2859 | 0.1124 |
| 512×512 | 188.6107 | 129.9131 | 131.7032 | 1.1289 | 1.0401 | 0.6058 |

At 512×512, tiled GPU total is 1.04 ms versus 131.70 ms for CPU transpose plus
multiply. The first complete GPU setup/upload/execute/readback is about
79 ms. Warm results must not be used as cold-start results.
Tiling helps the resident path. At 64×64, GPU naive total is 0.135 ms and tiled total is 0.123 ms. Transfer and call overhead can outweigh that gain.

## Other reference workloads

SAXPY computes `C[i] = 2*A[i] + B[i]`. Dot product reduces `A[i]*B[i]` in
256-element workgroups, reads partial sums, and finishes their sum on CPU.
The dot GPU total includes that final sum. Its resident column measures only
partial reduction, not a complete device-resident dot product.

| Workload | Elements | Native CPU ms | GPU total ms | Resident GPU ms |
| --- | ---: | ---: | ---: | ---: |
| saxpy | 1,024 | 0.0013 | 0.1196 | 0.0342 |
| saxpy | 16,384 | 0.0140 | 0.1387 | 0.0365 |
| saxpy | 1,000,000 | 0.7913 | 1.4054 | 0.0522 |
| saxpy | 10,000,000 | 7.7447 | 13.1785 | 0.6353 |
| dot | 1,024 | 0.0009 | 0.1158 | 0.0355 |
| dot | 16,384 | 0.0105 | 0.1310 | 0.0326 |
| dot | 1,000,000 | 0.6441 | 0.9878 | 0.0840 |
| dot | 10,000,000 | 6.2747 | 8.6926 | 0.5568 |

Fresh-input SAXPY and dot product favor native CPU at every sampled size.
Resident GPU work wins at the large sizes, but that excludes transfers.
Keeping data on the device has more value than tuning these small kernels.

For the ten-million-element reference cases:

| Workload | Upload ms | Dispatch ms | Download ms | Final CPU sum ms | Device buffer MiB | First setup ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| saxpy | 8.018 | 0.692 | 4.486 | 0.000 | 152.59 | 103.80 |
| dot | 7.983 | 0.626 | 0.057 | 0.025 | 76.59 | 96.68 |

Reference GPU execution reuses input, output, uniform, and readback buffers.
It uses about 565 Go bytes/10 allocations per resident call and 1,264 bytes/30
allocations per full call. Native CPU measurements report zero allocations per call. Some matrix
variants report 5 bytes per call from an amortized one-time allocation. Device-buffer bytes are fixed allocated storage, not a native-driver
memory peak. They exclude driver objects and transient queue upload payloads.
The reference tests do not run through the plugin's instance budget; they
cannot establish plugin admission limits or fallback behavior for these shaders.
Host input arrays, expected output, and Wasm memory are allocated once outside
warm timing. Wasm reserves two array regions for dot, three for SAXPY, and
four for matrices. The dot case avoids 80 MB of unused Wasm memory at ten
million elements. Dot uses one bounded output array of partial sums.

The CPU fixtures are checked against an independent F64 host calculation for
small matrices and vectors. Hardware tests change matrix input while reusing
resources and test vector counts 1, 255, 256, 257, and 16,384. Full result arrays
are checked against native Wago after every timed end-to-end GPU execution;
resident outputs are checked after the timed sequence. Comparisons require
finite values and use `8e-6 * max(1, abs(CPU))`. These bounded inputs do not
establish strict IEEE arithmetic for arbitrary inputs.

## Mandelbrot follow-up

Same settings before and after this change: 32 iterations, loaded module/device,
new WASI instance per render, output to `io.Discard`, three samples of three
renders, `GOMAXPROCS=1`, stage profiling disabled. The baseline is commit
`91dcd60`. These are separate runs from the earlier report.

| Image/path | Before ms | After ms | Before allocations | After allocations |
| --- | ---: | ---: | ---: | ---: |
| 512x512/cpu | 5.929 | 5.527 | 429 | 429 |
| 512x512/fallback | 25.048 | 22.286 | 1,155 | 763 |
| 512x512/gpu | 31.001 | 29.446 | 2,449 | 2,381 |
| 1920x1080/cpu | 45.537 | 43.307 | 429 | 429 |
| 1920x1080/fallback | 198.465 | 181.572 | 1,155 | 763 |
| 1920x1080/gpu | 248.803 | 199.184 | 2,449 | 2,381 |
| 2560x1440/cpu | 78.126 | 74.683 | 429 | 429 |
| 2560x1440/fallback | 427.349 | 393.347 | 1,155 | 763 |
| 2560x1440/gpu | 439.010 | 365.723 | 2,449 | 2,381 |

CPU fallback allocations fall from 1,155 to 763 per render. GPU allocations
fall from 2,449 to 2,381. This paired run also has lower times for all paths,
including the unchanged direct CPU path. Clock and temperature differences
can affect the comparison. Direct CPU remains faster for all
Mandelbrot sizes. The 1440p GPU tracked peak remains 154.70 MiB; this change
adds no buffer capacity. CPU loops remain linear per iteration. Mandelbrot
still transfers and seeds 900 MiB of orbit data per 1440p, 32-pass render.
Closing each per-render instance invalidates its groups. The two-entry cache
reuses recurring bindings within a render; it does not remove setup for a new
instance. Transfers and seeds remain the main costs.

Both full-HD and 1440p 64-iteration fallback images match direct CPU exactly.
Both GPU images completed all 64 passes without fallback. GPU float arithmetic
is relaxed; the image check records the boundary-pixel differences separately.

## Evidence and commands

No timed benchmark processes ran together. GPU clocks, CPU frequency, and
thermal state were not fixed. These are experimental samples on one laptop.
Host stages include driver/API work and completion waits. Device timestamps
are unavailable; `compute-us/op` must not be described as pure GPU kernel time.
No result is from a software adapter.

- [Compiled vector samples](results/common-compiled-bench.jsonl).
- [Arithmetic size test](results/common-polynomial-threshold.jsonl).
- [Reference benchmark samples](results/common-reference-bench.txt).
- [Mandelbrot before](results/perf-common-before.txt) and [after](results/perf-common-after.txt).
- [CPU suite](results/perf-common-cpu-tests.txt), [CPU race tests](results/perf-common-cpu-race.txt), [real-GPU race suite](results/perf-common-hardware-race.txt), and [cgo pointer checks](results/perf-common-cgocheck.txt).
- [Large 64-pass image hashes and differences](results/perf-common-images.json).

```sh
CGO_ENABLED=0 go test -count=1 ./...
WAGO_GPU_TEST=1 go test -race -count=1 -tags webgpu ./...

GOMAXPROCS=1 go run -tags webgpu ./cmd/buffer-bench \
  -kernels=twice,square,copy,polynomial -repeats=3 > vectors.jsonl
GOMAXPROCS=1 go run -tags webgpu ./cmd/buffer-bench \
  -kernels=polynomial -sizes=2048,4096,8192,12288,16384,32768 -repeats=9 > threshold.jsonl

# CPU references run without GPU hardware. GPU cases skip unless required.
CGO_ENABLED=0 GOMAXPROCS=1 go test . -run '^$' \
  -bench '^BenchmarkCommon/' -benchtime=3x -count=3 -benchmem
WAGO_GPU_TEST=1 GOMAXPROCS=1 go test -tags webgpu . -run '^$' \
  -bench '^BenchmarkCommon/' -benchtime=3x -count=3 -benchmem
```

Build the patched native library as described in the root README before GPU
commands. Fixture rebuilds use `go generate ./internal/fixtures` and WABT.
CPU-only tests use the checked-in Wasm and need no fixture compiler.

## Next three improvements

1. Add a bounded loop/control-flow subset so Mandelbrot can keep orbit and
   escape state on the GPU. Keep every unsupported case on explicit fallback.
2. Investigate batching uploads and commands with owned staging storage and
   one completed error protocol. Do not remove waits or error scopes without
   proving cancellation, lifetime, and commit safety.
3. Add restricted matrix indexing and reduction support only with a separate
   contract and correctness tests. Then compare compiled-Wasm results with
   these reference ceilings and with an optimized CPU matrix implementation.
