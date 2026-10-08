# Compiler and transfer follow-up

Measured on 2026-10-08 against commit `07db1f4`. This report follows
[the first compiler changes](COMPILE_PERFORMANCE.md). Wago and the public plugin
ABI are unchanged. All timed GPU operations ran on a hardware GPU and checked
their results. No authored reference shader is used in these new cases.

The main gain is avoiding a full upload after a small CPU write. A one-element
patch in a ten-million-element input now takes 78.699 µs, down from 4,058.430 µs.
A call with four 1,024-element inputs takes 101.978 µs, down from 167.664 µs.
Fresh ten-million-element inputs still take about 12.4 ms, including readback
and guest copies. The new changes do not make that simple multiply-by-two job
faster than direct native Wago CPU execution.

Translation of the twice kernel takes 3.285 µs, down from 5.756 µs.
Mandelbrot-step translation takes 4.753 µs, down from 7.055 µs. Complete native
module compilation changed little. The unchanged native compiler, source
snapshots, hashing, and device setup still limit complete setup performance.

## Changes and safety rules

### Classify imports once

Decode each intrinsic's name, signature, scalar type, and element type once.
Keep them in an eight-byte numeric import record. Compare borrowed name bytes
while decoding. Do not retain strings for unrelated imports. Size metadata
slices from bounded section counts instead of growing them repeatedly.

Lowering tests the stored flags. It does not repeat string prefix/suffix checks,
signature-array comparisons, or element-type searches for every call. A known
bad signature remains a typed invalid-contract error when called. An unrelated
or unknown call remains unsupported. An unused bad import does not reject an
otherwise accepted kernel. Wago still validates the complete module.

The earlier immutable value records, bounded emission depth, saved-load rules,
private scratch, and final-source identity checks remain. There is no new
module cache, shared arena, or relaxed contract check. The Mandelbrot decoder's
Go allocation cost falls from 4,880 to 2,648 bytes per decode.

### Upload a bounded dirty range

After a completed GPU upload, keep one interval of CPU writes in each buffer.
Offsets are elements. Several writes extend its minimum and maximum. This costs
eight bytes per buffer and constant work per write. It does not scan the buffer
or allocate a range list. Gaps inside the interval are uploaded too.

A new buffer or unknown GPU base requires a full upload. A range becomes current
only after successful completion. Failed transfers still quarantine the device.
Version changes, full-set lost-content repair, and output commit rules remain.
Narrow types expand only the dirty interval into the existing conversion scratch.
The physical upload uses four-byte GPU words; logical counters use canonical
storage bytes.

The patch benchmark first uploads the large input outside timing. Each timed
operation changes one element, dispatches one invocation into a separate
one-element output, and reads that result. The payload falls from 40,000,000
bytes to four bytes in the ten-million-element case. It does not measure a
full-size output, seed, or readback. A small read from a large GPU-current output
can still require full readback under the current CPU mirror contract.

### Combine inputs with dispatch

When stage profiling is disabled, the native backend can queue F32/I32/U32
uploads, output seeds, parameter upload, and compute in one ordered submission.
It uses one completed dispatch wait. Narrow conversion and backends without
this capability keep the synchronous upload path. Profiled calls keep separate
waits so their stage times remain meaningful.

Allocate and reserve all resources and queue payloads before queueing. Source
bytes belong to the plugin and remain protected by its lock. No guest-memory
borrow crosses a GPU wait. Pending native payloads remain charged on failure.
All output versions still commit together after all checks pass. A cancellation
can keep an unchanged completed input mirror as cache data; it cannot publish
an uncommitted output.

Combining uploads can raise peak tracked memory because several queue payloads
are live together. Those bytes are reserved together, including the parameter
payload. The operation can return LIMIT_EXCEEDED before submission. This trades
bounded peak memory for fewer waits. Go allocations fall, but the small fresh
case uses 80 more Go bytes per operation. The profiled path uses 232 more bytes.
The tables report these increases. The 64-pass 1440p check reached 196.9 MiB
of tracked storage, up from 154.7 MiB in the previous check. Full HD reached
110.7 MiB, up from 87.01 MiB. Both fit the default 256 MiB instance limit.
These peaks include known native queue payloads; they do not include unknown
driver overhead.

### Use map completion for staging readback

Remove the separate queue-completion wait before mapping the staging buffer.
Successful mapping waits for preceding GPU use of that buffer, including its
submitted copy. See [WebGPU buffer mapping](https://www.w3.org/TR/webgpu/#buffer-mapping).
Checked encoding and submission error scopes still complete. Map status,
device loss, cancellation, mapped range, and the final CPU copy commit are still
checked. Map completion does not replace error reporting. The same duplicate
wait was removed from the legacy scalar readback path.

## Method

Baseline commit: `07db1f4747991b9f6d4681f4d292398cda4c511e`. The same new
`marshal_benchmark_test.go` was added to the archived baseline solely to measure
its existing behavior. The compiler harness already existed at that commit.
Both binaries use the same Wago dependency, native library, fixtures, Go version,
and configuration. The baseline arithmetic and buffer code were unchanged.

Five process pairs ran sequentially. Pair order alternated. CPU cases used 500
operations per sample, marshalling cases 20, and pipeline cases 100. Tables show
the median of five process means, rather than a pooled per-call median. Each
marshalling case has two warm operations outside timing. Runtime, device,
module, pipeline, and buffers stay live. Setup, initial allocation, and final
result checks are outside timing. Timed work includes Wasm invocation, checked
guest copies, plugin CPU copies, GPU transfers, dispatch, waits, and readback.

`GOMAXPROCS=1`, default Go GC, normal clocks, and uncleared driver caches. No
builds or tests ran during the timed pairs. AMD Ryzen 7 8845HS, NVIDIA RTX 4060
Laptop GPU with 8 GiB, Vulkan driver 550.163.01, Debian 13.7, Linux 6.12.111,
Go 1.27.1, amd64. CPU binaries use CGO_ENABLED=0; GPU binaries use `-tags webgpu`.
Go bytes are cumulative allocation, not peak live memory. Native driver memory
is outside these Go figures and outside known buffer accounting.

Settings, binary hashes, sample minima/maxima, and medians are in
[settings](results/round2-settings.json) and [summary](results/round2-summary.json).
Trailing whitespace was removed from logs. Raw files are `results/round2-paired-{cpu,marshal,pipeline}-{before,after}-{1..5}.txt`.

## Compiler measurements

| Work | Before µs | After µs | Before / after | Go allocations | Go bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| FourKernels / Decode | 3.333 | 2.385 | 1.40× | 38 → 30 | 5,624 → 3,224 |
| FourKernels / StandaloneKernel | 5.756 | 3.285 | 1.75× | 40 → 32 | 8,824 → 6,424 |
| FourKernels / PrepareSource | 31.357 | 30.001 | 1.05× | 79 → 71 | 41,706 → 39,306 |
| FourKernels / NativeOnly | 224.239 | 228.465 | 0.98× | 454 → 454 | 229,748 → 229,748 |
| FourKernels / NativeWithPlugin | 245.689 | 242.019 | 1.02× | 523 → 515 | 267,418 → 265,018 |
| Polynomial / Decode | 3.967 | 2.607 | 1.52× | 38 → 30 | 5,624 → 3,224 |
| Polynomial / StandaloneKernel | 14.309 | 12.851 | 1.11× | 46 → 38 | 27,512 → 25,112 |
| Polynomial / PrepareSource | 24.669 | 24.698 | 1.00× | 64 → 56 | 37,359 → 34,959 |
| Polynomial / NativeOnly | 230.352 | 224.925 | 1.02× | 454 → 454 | 229,749 → 229,748 |
| Polynomial / NativeWithPlugin | 248.903 | 251.154 | 0.99× | 514 → 506 | 263,233 → 260,833 |
| Mandelbrot / Decode | 4.631 | 2.485 | 1.86× | 43 → 25 | 4,880 → 2,648 |
| Mandelbrot / StandaloneKernel | 7.055 | 4.753 | 1.48× | 45 → 27 | 8,464 → 6,232 |
| Mandelbrot / PrepareSource | 23.307 | 23.032 | 1.01× | 63 → 45 | 37,511 → 35,279 |
| Mandelbrot / NativeOnly | 1,092.363 | 1,116.326 | 0.98× | 707 → 707 | 591,254 → 591,254 |
| Mandelbrot / NativeWithPlugin | 1,093.494 | 1,074.726 | 1.02× | 766 → 748 | 624,880 → 622,648 |
| LargeCustom / Decode | 4.762 | 2.877 | 1.66× | 38 → 30 | 5,624 → 3,224 |
| LargeCustom / StandaloneKernel | 5.837 | 4.018 | 1.45× | 40 → 32 | 8,824 → 6,424 |
| LargeCustom / PrepareSource | 1,073.685 | 1,058.479 | 1.01× | 81 → 73 | 4,247,115 → 4,244,710 |
| LargeCustom / NativeOnly | 333.298 | 331.507 | 1.01× | 457 → 457 | 2,327,012 → 2,327,012 |
| LargeCustom / NativeWithPlugin | 2,282.922 | 2,305.732 | 0.99× | 527 → 519 | 6,570,051 → 6,567,651 |

These cases and limits are defined in the earlier compiler report. The
native-only control uses unchanged Wago code. Its variation is about 1–2%.
Similar small changes in whole-module compilation are not evidence of a major
speedup. The polynomial preparation and large-source full compilation show no
clear improvement. Lower allocation remains useful, but does not remove their
largest costs. Legacy scalar compilation was unchanged: 4.501 → 4.420 µs,
18 allocations and 6,944 Go bytes in both versions.

## GPU marshalling measurements

| Work | Before µs | After µs | Before / after | Go allocations | Go bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh 1,024; profile false | 93.675 | 78.111 | 1.20× | 44 → 37 | 2,904 → 2,984 |
| Fresh 1,024; profile true | 85.500 | 83.298 | 1.03× | 44 → 41 | 2,904 → 3,136 |
| Fresh 16,384; profile false | 103.625 | 93.046 | 1.11× | 44 → 37 | 2,904 → 2,984 |
| Fresh 16,384; profile true | 102.386 | 100.626 | 1.02× | 44 → 41 | 2,904 → 3,136 |
| Fresh 1,000,000; profile false | 1,324.499 | 1,328.903 | 1.00× | 44 → 37 | 2,904 → 2,984 |
| Fresh 1,000,000; profile true | 1,324.261 | 1,325.898 | 1.00× | 44 → 41 | 2,904 → 3,136 |
| Fresh 10,000,000; profile false | 12,583.614 | 12,441.788 | 1.01× | 56 → 49 | 3,272 → 3,352 |
| Fresh 10,000,000; profile true | 13,011.257 | 12,596.018 | 1.03× | 56 → 53 | 3,272 → 3,504 |
| One-element patch in 1,000,000-element input | 501.113 | 79.094 | 6.34× | 50 → 43 | 3,080 → 3,160 |
| One-element patch in 10,000,000-element input | 4,058.430 | 78.699 | 51.57× | 50 → 43 | 3,080 → 3,160 |
| Four inputs, 1,024 elements each | 167.664 | 101.978 | 1.64× | 77 → 58 | 4,808 → 4,432 |
| Four inputs, 1,000,000 elements each | 2,821.253 | 2,840.860 | 0.99× | 77 → 58 | 4,808 → 4,432 |

Fresh input uses the compiled Wasm multiply-by-two kernel. The four-input case
sums four arrays into its fourth buffer; that READ_WRITE output needs an old-value
seed. All cases check dispatch status and output. A CPU fallback status fails
these hardware benchmarks. The patch alternates input values to avoid measuring
unchanged contents.

The small four-input gain comes from fewer submissions and waits. At one million
elements per input, transfer volume dominates and that gain disappears.
The fresh full-array million-element results also show no clear gain. Stage
profiling is a control with its own waits; do not compare its host stage values
to the combined path's unavailable stages. Both paths keep complete counters.

## Pipeline controls

| Work | Before µs | After µs | Before / after | Go allocations | Go bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Twice | 128.448 | 129.687 | 0.99× | 6 → 6 | 177 → 177 |
| 32-step arithmetic | 380.729 | 377.000 | 1.01× | 6 → 6 | 177 → 177 |
| Mandelbrot step | 217.776 | 195.672 | 1.11× | 6 → 6 | 177 → 177 |

Generated WGSL is unchanged by this follow-up: 515 bytes for twice, 3,649 for
32-step arithmetic, and 1,007 for the Mandelbrot step. There is no new shader
optimizer here. Treat timing changes in this table as driver/cache variation,
not an attributed pipeline speedup. Device creation remains excluded.

## Checks

The CPU-only suite, real-GPU race suite, strengthened cgo-pointer suite, public
ABI/TinyGo checks, vet, bounded compiler fuzzing, and large examples are checked
separately. New tests cover dirty unions across bulk and scalar writes, failed
range checks, unchanged-input reuse, narrow conversion, unknown GPU bases,
preflight resource failure, uncertain payload retention, cancellation after
completion, and complete cleanup. Import tests cover namespace, name, signature,
and unused malformed-signature classification.

See [CPU tests](results/round2-cpu-tests.txt),
[GPU race tests](results/round2-gpu-race-tests.txt),
[cgo checks](results/round2-cgocheck-tests.txt),
[fuzz log](results/round2-fuzz.txt), and
[specification checks](results/round2-spec.txt).
The bounded CPU compiler fuzz run completed 304,316 inputs in 31 seconds
without a reported failure. The specification and TinyGo checks passed all
11 checks. Vet passed. Narrow dirty uploads also ran on the real GPU for U8,
U16, and F16, with exact output and unchanged-element checks.

The [64-pass full-HD and 1440p checks](results/round2-mandelbrot-check.txt)
completed all GPU passes. CPU fallback matched direct CPU bytes exactly. GPU
output still differs at 198 full-HD pixels and 366 1440p pixels under the stated
relaxed-float contract. These single-render samples are correctness checks,
not a replacement for the paired measurements or earlier repeated render data.

[All four vector kernels](results/round2-vector-check.jsonl) passed direct
Wago CPU comparisons at 1,024, 16,384, 1,000,000, and 10,000,000 elements. Each
case checked first, repeated, resident, and prefix dispatch. The one-repeat
settings are correctness evidence, not a new crossover study. The largest
vector case reached 240,000,032 tracked bytes (228.9 MiB), within the default
256 MiB limit.

## Remaining work

1. Keep a runtime, module, and pipeline live across jobs. This avoids device
   startup and compilation rather than speeding those costs up. A generic
   read-only Wago source observer could avoid unchanged transform snapshots,
   but needs final-source identity and verified artifact adoption. No Wago
   change was made in this work.
2. Keep intermediate buffers on the GPU. The public buffer API already permits
   this. Reading large intermediates after every pass still forces full readback.
   Extending the compiler to accept complete loops or reductions needs separate
   contract and numerical checks. Matmul remains an authored reference case.
3. Measure upload staging reuse and device timestamp queries. Preserve bounded
   memory, callback ownership, completed error scopes, and transactional outputs.
   Hardware coverage still comes from this one NVIDIA Vulkan system.

## Reproduce

```sh
GOMAXPROCS=1 CGO_ENABLED=0 go test . -run '^$' \
  -bench '^BenchmarkCompilePath$' -benchtime=500x -count=5 -benchmem
WAGO_GPU_TEST=1 GOMAXPROCS=1 go test -tags webgpu . -run '^$' \
  -bench '^BenchmarkBufferMarshal$' -benchtime=20x -count=5 -benchmem
WAGO_GPU_TEST=1 GOMAXPROCS=1 go test -tags webgpu . -run '^$' \
  -bench '^BenchmarkShaderBuild$' -benchtime=100x -count=5 -benchmem
```

For paired measurements, build archived commit `07db1f4` with the new marshalling
harness. Build both versions before timing. Alternate the binaries with the
same `-test.*` flags and set `-test.count=1`. Runtime creation, device startup,
and new-module first use are separate from these retained-resource measurements.
