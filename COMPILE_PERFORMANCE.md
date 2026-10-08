# Compilation performance

This is the first compiler change. The next measurements and import-record
changes are in [the compiler and transfer follow-up](MARSHAL_PERFORMANCE.md).

Measured on 2026-10-08. The plugin now uses a single forward pass with compact
symbolic values, as in the Valent-Block design. The supported Wasm subset and
public plugin API are unchanged. Wago is unchanged.

The largest translation gains are in arithmetic kernels. Translation of the
32-step kernel is 62% faster, and uses 46 Go allocations instead of 636.
Preparing the module with four kernels is 51% faster. Complete native Wago
compilation with the plugin is 19% faster for that module. A module with a
2 MiB custom section compiles 28% faster with the plugin.

These changes do not remove GPU device startup. Earlier full-setup measurements
were about 82–91 ms for the small examples. The new compilation measurements
exclude device opening and closing. Do not apply the compiler speedups to
those full-setup times.

## Design

The implementation uses the parts of Valent-Block that suit the restricted
WGSL compiler:

- Decode module metadata once for the selected buffer kernels. Use a fixed
  section-order table. Resolve intrinsic names once per imported function.
- Keep eight-byte immutable value records on the operand stack and in locals.
  Constants retain their bits. Handles and the invocation index keep separate
  categories. A local assignment changes its current record, not an earlier
  stack value.
- Record pure arithmetic as compact nodes. Emit a value when a consumer needs
  it. Emit each node at most once. A value reused twice at each step does not
  cause exponential expression expansion.
- Condense deferred chains at depth six. This bounds recursive emission. Use
  numeric indexes, so slice growth cannot invalidate references to nodes.
- Materialize a buffer load at its instruction point. Later stores cannot
  change its saved result. Float copies keep integer bits; arithmetic uses
  floating-point temporaries under the existing relaxed-float contract.
- Reuse private scratch between selected functions in one compilation. Small
  stacks, locals, nodes, and output use inline storage. Larger functions use
  bounded slice growth. Copy final shader text into owned storage before reset.
- Emit numbers and text directly. Avoid formatted string allocation for each
  successful instruction. Use fixed eight-slot arrays for binding/use tracking.
- Hash the source once for both plugin interfaces. Skip legacy scalar decoding
  when no legacy kernel export is configured.

No full SSA pass, control-flow graph, register allocator, cross-module cache,
shared mutable arena, or unbounded scratch pool was added. The compiler permit
still serializes lowering within one plugin. Separate plugins have separate
scratch. Existing source, local, instruction, stack, and output limits remain.

The legacy scalar compiler also emits numbers directly. It keeps its existing
instruction-driven temporary model. It does not need the buffer compiler's
mutable-local and saved-load machinery.

## Method

The baseline is commit `a6c242b1ef7f554b2338a4562684b58d6403bfd4`, plus the same
new benchmark harness. The baseline binaries were built before the changes.
Both versions use the same Wago dependency, fixtures, configuration, and Go
version. Binary hashes and settings are in [compile-settings.json](results/compile-settings.json).

Five process pairs ran in sequence. Pair order alternated before/after and
after/before. CPU cases used 500 operations per sample. Native GPU pipeline
cases used 100 builds per sample. The tables show the median of the five
process-average measurements. Builds and correctness tests did not run during
these measurements. `GOMAXPROCS=1`; Go GC uses its default settings. GPU clocks,
CPU clocks, and temperature were not fixed. Driver caches were not cleared.
Small timing differences can be noise; the native-only control shows this.

Hardware: AMD Ryzen 7 8845HS, NVIDIA RTX 4060 Laptop GPU (8 GiB), Vulkan driver
550.163.01, Debian 13.7, Linux 6.12.111, Go 1.27.1, amd64. CPU benchmarks have
`CGO_ENABLED=0`. GPU builds use `-tags webgpu` and the checked local native
binding. Go allocation figures exclude native driver memory. Bytes per operation
measure cumulative Go allocation, not peak live memory.

The cases are:

| Case | Source bytes | Work |
| --- | ---: | --- |
| FourKernels | 2,378 | Four selected buffer kernels: twice, square, copy, 32-step arithmetic |
| Polynomial | 2,378 | The same source; only the 32-step arithmetic kernel selected |
| Mandelbrot | 11,375 | One selected two-output orbit-step kernel; the guest keeps the loop |
| LargeCustom | 2,099,536 | Four-kernel source plus a valid 2 MiB custom section |
| LegacyHeavy | 862 | Legacy scalar 32-step arithmetic function |

`Decode` measures only the bounded plugin decoder. `StandaloneKernel` measures
`CompileBufferWGSL`, including decode, for the first configured kernel: twice
in FourKernels/LargeCustom, polynomial in Polynomial, step in Mandelbrot.

`PrepareSource` measures `Runtime.PrepareCompile` and `PreparedCompile.Close`.
It includes Wago source snapshots, plugin contract inspection, all configured
kernel translations, the source hash, and bounded pending-record eviction.
The device is disabled. Closed preparations still have no disposal hook, so
these runs exercise the existing 64-record/16-translation eviction policy.

`NativeOnly` and `NativeWithPlugin` measure fresh `Runtime.Compile` plus module
closure on a retained runtime. The latter has GPU execution disabled but still
performs contract verification and translation. Device/pipeline creation and
runtime creation/closure are excluded. These are complete native CPU module
compilations, not GPU end-to-end startup measurements.

## Decoder and translation

| Module / step | Before µs | After µs | Change | Before → after allocations | Before → after bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| FourKernels / Decode | 5.543 | 3.279 | 40.8% faster | 92 → 38 | 8,248 → 5,624 |
| FourKernels / StandaloneKernel | 9.229 | 5.689 | 38.4% faster | 124 → 40 | 11,032 → 8,824 |
| Polynomial / Decode | 6.458 | 4.207 | 34.9% faster | 92 → 38 | 8,248 → 5,624 |
| Polynomial / StandaloneKernel | 39.508 | 14.883 | 62.3% faster | 636 → 46 | 42,058 → 27,512 |
| Mandelbrot / Decode | 8.210 | 4.477 | 45.5% faster | 117 → 43 | 8,816 → 4,880 |
| Mandelbrot / StandaloneKernel | 17.942 | 6.719 | 62.6% faster | 210 → 45 | 14,752 → 8,464 |
| LargeCustom / Decode | 6.736 | 4.775 | 29.1% faster | 92 → 38 | 8,248 → 5,624 |
| LargeCustom / StandaloneKernel | 10.598 | 5.937 | 44.0% faster | 124 → 40 | 11,032 → 8,824 |

| Module / step | Before µs | After µs | Change | Before → after allocations | Before → after bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| LegacyHeavy / CompileWGSL | 15.870 | 4.530 | 71.5% faster | 93 → 18 | 18,592 → 6,944 |

## Preparation and complete native compilation

| Module / step | Before µs | After µs | Change | Before → after allocations | Before → after bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| FourKernels / PrepareSource | 64.996 | 32.065 | 50.7% faster | 765 → 79 | 60,027 → 41,706 |
| FourKernels / NativeOnly | 240.557 | 243.197 | 1.1% slower | 454 → 454 | 229,748 → 229,748 |
| FourKernels / NativeWithPlugin | 310.693 | 252.974 | 18.6% faster | 1209 → 523 | 285,741 → 267,418 |
| Polynomial / PrepareSource | 52.442 | 25.748 | 50.9% faster | 656 → 64 | 51,920 → 37,359 |
| Polynomial / NativeOnly | 239.758 | 228.017 | 4.9% faster | 454 → 454 | 229,749 → 229,748 |
| Polynomial / NativeWithPlugin | 294.873 | 265.494 | 10.0% faster | 1105 → 514 | 277,796 → 263,233 |
| Mandelbrot / PrepareSource | 41.412 | 24.316 | 41.3% faster | 229 → 63 | 43,816 → 37,511 |
| Mandelbrot / NativeOnly | 1114.869 | 1127.823 | 1.2% slower | 707 → 707 | 591,255 → 591,253 |
| Mandelbrot / NativeWithPlugin | 1161.483 | 1127.501 | 2.9% faster | 932 → 766 | 631,200 → 624,880 |
| LargeCustom / PrepareSource | 2063.977 | 1129.819 | 45.3% faster | 767 → 81 | 4,265,429 → 4,247,110 |
| LargeCustom / NativeOnly | 345.515 | 347.926 | 0.7% slower | 457 → 457 | 2,327,012 → 2,327,012 |
| LargeCustom / NativeWithPlugin | 3289.544 | 2363.804 | 28.1% faster | 1213 → 527 | 6,588,387 → 6,570,051 |

Native-only compilation uses unchanged dependency code. Its small timing
changes are control variation. The Mandelbrot native compiler takes about
1.1 ms; faster plugin translation changes that total by only about 3%.
The large custom section is skipped by the decoder without copying its payload,
but whole-source hashing and Wago snapshots still process that payload.

## Native GPU pipeline creation

These cases build a real native shader module and compute pipeline from the
generated WGSL, then close it. The device stays open. The plugin pipeline cache
is bypassed so each iteration builds a new pipeline resource. Driver caches can
still be warm. The measurements include native validation/error checks and
pipeline closure. They do not include buffer transfers or kernel execution.

| Generated kernel | Before µs | After µs | Change | New WGSL bytes | Go allocations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Twice | 144.496 | 138.470 | 4.2% faster | 515 | 6 → 6 |
| 32-step arithmetic | 495.640 | 387.553 | 21.8% faster | 3649 | 6 → 6 |
| Mandelbrot orbit step | 240.378 | 191.635 | 20.3% faster | 1007 | 6 → 6 |

This is repeated pipeline construction, not a cold driver-cache benchmark.
Device timestamps are not used. Pipeline correctness is checked separately by
running the shaders on the GPU. First device startup remains in the
[earlier example measurements](examples/BENCHMARKS.md).

## Contract checks

The CPU-only suite, real-GPU race suite, and strengthened cgo-pointer checks
pass. Specification and TinyGo declaration checks pass (11 checks). `go vet`
passes. A 30-second CPU-only decoder/compiler fuzz run completed 439,769 inputs
without a reported failure.

New tests check scratch reuse across kernels and repeats, unchanged owned
shader strings after reset, linear output for 256 shared arithmetic nodes,
local snapshots, zero-initialized float locals, and a saved load consumed after
a store. The last three run on the real GPU over 65 elements. Existing tests
also cover exact float bits used for both arithmetic and copy, narrow integer
storage, F16 conversion, seed/read-before-write rules, unsupported instructions,
metadata eviction, source mismatch, fallback, cancellation, and cleanup.

The 64-pass full-HD and 1440p Mandelbrot programs were checked again. CPU
fallback still matches direct CPU output exactly. All GPU passes succeeded.
The GPU still differs at 198 full-HD pixels and 366 1440p pixels under the
relaxed-float contract. [The check log](results/compile-mandelbrot-check.txt)
also has one timed render per case; those single samples do not replace the
earlier repeated execution measurements.

All four vector kernels also passed direct-Wago-CPU comparisons on the GPU at
1,024, 16,384, 1,000,000, and 10,000,000 elements. Each case includes first,
repeated, resident, and prefix dispatch checks. [The vector check data](results/compile-vector-check.jsonl)
uses one repeated sample per case and is correctness evidence for this change.

Resource commit, checked guest storage, device error reporting, callback
ownership, and CPU fallback rules are unchanged. The compiler does not reorder
buffer operations or reassociate arithmetic. Dead pure arithmetic can remain
unemitted; its instruction types and contract classification are still checked.

## Remaining costs and next steps

1. **Wago source observation.** The pinned source-transform API takes a complete
   snapshot before hooks and after each hook, including unchanged output.
   A transform hook also makes `PreparedCompile.Cacheable()` false. These are
   Wago ownership rules. Returning the same slice or `nil` does not avoid them.
   The plugin keeps those rules and the final compiled-source digest check.
   A generic read-only source-inspection hook with final-source identity could
   reduce snapshots and preserve cache eligibility. It would need explicit
   ownership, transform ordering, and artifact-adoption verification. No Wago
   API change is required for this implementation, and no dependency was changed.
2. **GPU device startup.** Retain one runtime/device for repeated jobs, as the
   reuse-module examples already do. Opening a new device for each small job
   still costs much more than these compiler gains. A separate lazy-device
   policy could help CPU-only or below-threshold jobs, but must preserve start
   functions, module closure, error reporting, and retry rules.
3. **Shader compilation.** The existing pipeline cache reuses validated shaders
   in a runtime. Measure cold driver-cache builds separately before adding
   persistent caches or more code-generation rules. Matmul, branches, and the
   complete Mandelbrot loop still need compiler extensions; authored reference
   shaders remain separate from supported Wasm translation.

## Reproduce

Run from this checkout. Native GPU tests require the pinned native library and
an actual GPU. See [build instructions](README.md#build-and-test).

```sh
GOMAXPROCS=1 CGO_ENABLED=0 go test . -run '^$' \
  -bench '^BenchmarkCompilePath$' -benchtime=500x -count=5 -benchmem

WAGO_GPU_TEST=1 GOMAXPROCS=1 go test -tags webgpu . -run '^$' \
  -bench '^BenchmarkShaderBuild$' -benchtime=100x -count=5 -benchmem

CGO_ENABLED=0 go test ./...
WAGO_GPU_TEST=1 go test -race -tags webgpu ./...
WAGO_GPU_TEST=1 GOEXPERIMENT=cgocheck2 go test -tags webgpu ./...
CGO_ENABLED=0 GOMAXPROCS=2 go test . -run '^$' \
  -fuzz '^FuzzBufferCompiler$' -fuzztime=30s
python3 spec/check_spec.py --tinygo
go vet -tags webgpu ./...
```

For an exact comparison, build the baseline with `compile_benchmark_test.go`
from this change, then build the new version. Build both CPU and GPU binaries
before timing. Alternate the two binaries per process pair with the flags above,
using `-test.run`, `-test.bench`, `-test.benchtime`, `-test.count=1`, and
`-test.benchmem`. The baseline report-metric placement was corrected in the
new harness, so only the new shader byte metric is printed. That correction
is outside the measured build loop.

[Median summary](results/compile-summary.json) and
[settings and binary hashes](results/compile-settings.json) are machine-readable.
Raw paired data is in `results/compile-paired-{cpu,gpu}-{before,after}-{1..5}.txt`.
The initial baseline files use a different iteration count and are not included
in the tables. Test logs: [CPU](results/compile-cpu-tests.txt),
[GPU race](results/compile-gpu-race-tests.txt),
[cgo checks](results/compile-cgocheck-tests.txt), and
[fuzz](results/compile-fuzz.txt).
