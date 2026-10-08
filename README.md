# wago-gpu

An optional Go plugin for Wago. It translates selected WebAssembly instructions
into WGSL and runs elementwise kernels on a GPU. Wago is an unchanged dependency.

The buffer API has 34 imports, CPU fallback, checked memory32/memory64/GC
transfers, typed storage, and explicit dispatch. Kernels can share buffers
without an intermediate guest transfer. The earlier scalar `wago_gpu.run` and
`run_batch` imports remain available.

`v0.0.0` is an experimental release. The ABI remains provisional. See the
[specification](BUFFER_API_PROPOSAL.md), [current results](BUFFER_REPORT.md), and
[canonical ABI table](spec/abi_v1.json).

## Measured findings

Real GPU execution passed the result checks. The new 32-step arithmetic kernel
is compiled from actual Wasm. It is faster on the GPU at the larger sizes.
In the buffer API tests, copy, twice, square, and Mandelbrot still favor direct
Wago CPU execution with fresh inputs. The scalar import demo has a different
CPU loop and transfer path; its results are reported separately.

Measured on 2026-10-08 with an NVIDIA RTX 4060 Laptop GPU, Vulkan driver
550.163.01, AMD Ryzen 7 8845HS, Debian 13/Linux 6.12, and Go 1.27.1.
The vector table uses nine repeated calls and `GOMAXPROCS=1`.
GPU totals include input upload, execution, readback, and guest copies.
Stage profiling is enabled.

| Compiled Wasm workload | Elements | Direct CPU ms | GPU total ms |
| --- | ---: | ---: | ---: |
| 32 multiply/add steps | 16,384 | 0.594 | 0.121 |
| 32 multiply/add steps | 1,000,000 | 18.791 | 1.436 |
| 32 multiply/add steps | 10,000,000 | 187.476 | 14.153 |
| Copy | 10,000,000 | 5.456 | 12.981 |
| Multiply by two | 10,000,000 | 6.144 | 13.092 |
| Square plus one | 10,000,000 | 7.820 | 13.476 |

A separate nine-sample size test found the first GPU win at 8,192 elements for
the 32-step kernel. That is a measured point on this machine, not a general
threshold. First-run device and compilation costs are separate from these
warm times.

The new common-workload suite also measures matrix multiplication, SAXPY,
and dot product. **Those GPU cases use authored reference WGSL. They do not
extend the plugin compiler.** Their CPU references are Wasm functions executed
by native Wago. At 512×512, tiled matrix GPU total was 1.040 ms. CPU transpose
plus multiply was 131.703 ms. The matrix loops are simple references, not BLAS.
Fresh-input SAXPY and dot product remained slower on the GPU at all tested
sizes. Device-resident timings show the value of avoiding transfers.

The latest changes reduce work while keeping the commit and lifetime rules:

- Cache at most two native binding groups. Release them before closing any
  referenced buffer or pipeline. Reuse stored buffer sizes.
- Seed only the unchanged output tail when covering stores need no old prefix.
  Keep full seeds for read-before-write kernels.
- Avoid timer allocation for CPU-only bulk copies and fallback with current
  CPU data. Check the same entry deadline before commit. Keep timers for GPU waits.
- Use two bulk imports in the benchmark CPU runner. Native Wasm performs
  the element loop. Partial output updates preserve their tails.

The recorded ten-million-element bulk CPU runner took 13.69 ms for twice and
16.75 ms for square. The earlier scalar-import runner took 10,418.39 ms and
15,357.40 ms. These are separate single samples on the same machine.
The change reduces host calls; both loops are linear. Wago and the public
plugin ABI remain unchanged.

All runnable examples were benchmarked, including start sections, both mixed
CPU/GPU buffer paths, TinyGo, F16, memory64, GC arrays, and both CLI demos.
See [all example results](examples/BENCHMARKS.md) for full setup times,
repeated times, allocations, first-run costs, raw data, and rerun commands.
The small examples are faster on the CPU. GPU full setup took about 82–91 ms.
With the module retained, the four-value TinyGo command took 0.025 ms on CPU
and 0.171 ms on GPU.

The latest Mandelbrot run uses the default 64 iterations and three samples
of three images. The module stays loaded. Each render creates a WASI instance
and includes setup, output to `io.Discard`, checks, and cleanup.
Stage profiling is disabled.

| Image | Direct CPU ms | Buffer CPU fallback ms | GPU ms |
| --- | ---: | ---: | ---: |
| 512×512 | 10.380 | 46.169 | 57.663 |
| 1920×1080 | 77.947 | 364.305 | 387.272 |
| 2560×1440 | 132.788 | 811.829 | 702.237 |

The direct CPU program is faster at all tested sizes. At 1440p the GPU is
faster than buffer CPU fallback. HD and 1440p images allocate 429 Go objects
on direct CPU, 923 with buffer fallback, and 4,109 on GPU per repeated render.
Clocks and temperature were not fixed. Device timestamps are unavailable.

The 1440p GPU tracked peak remains 154.70 MiB, below the 256 MiB instance limit.
The ten-million-element vector peak remains 228.88 MiB. Idle scratch storage
remains charged and can be released under pressure. Go allocation counts
exclude native driver allocations.

Both full-HD and 1440p 64-iteration fallback images match CPU exactly. The GPU
completed all 64 passes without fallback. Relaxed float arithmetic changes
198 full-HD pixels and 366 1440p pixels near the set boundary. Mandelbrot still
needs large orbit readbacks and seeds; moving its loop and escape check to the
GPU requires a compiler extension.

The CPU suite, real-GPU race suite, cgo pointer checks, and large image checks
passed. See [all example measurements](examples/BENCHMARKS.md) and
[the common benchmark report](COMMON_BENCHMARKS.md) for matrix,
SAXPY, dot, vector, transfer, first-run, allocation, memory, and crossover data.
It includes raw samples and rerun commands. The
[previous performance report](examples/mandelbrot/PERFORMANCE_FOLLOWUP.md) and
[first scratch report](examples/mandelbrot/PERFORMANCE.md) retain earlier results.

## Build and test

Go 1.25 or later is required. The tested Go version is 1.27.1.
The module path is `github.com/jtenner/wago-gpu`.

```sh
git clone https://github.com/jtenner/wago-gpu.git
cd wago-gpu
```

Run the commands below from this checkout. GPU builds need the local dependency
repairs selected by this repository's `go.mod`. Go does not apply a dependency's
`replace` directives to a consuming module. A GPU host in another module must
apply both replacements to this checkout as well; the unpatched upstream
binding is not supported. For a host module beside the checkout, use:

```sh
go get github.com/jtenner/wago-gpu@v0.0.0
go mod edit -replace github.com/oliverbestmann/webgpu=../wago-gpu/third_party/webgpu
go mod edit -replace github.com/oliverbestmann/webgpu/libs-linux=../wago-gpu/third_party/webgpu/libs-linux
```

Build the native library in the checkout with `./native/build.sh` before building
that host with `-tags webgpu`. The CPU path needs no native replacement build.

```sh
# No GPU, native library, or C compiler is needed.
CGO_ENABLED=0 go test ./...
go run ./examples/buffers -first-cpu -second-cpu
go run ./examples/tinygo -cpu

# Complete generated-Wasm tests also need wasm-tools on PATH.
go test -race ./...

# GPU build: Linux, Vulkan driver, C compiler, Rust/Cargo, and Git.
# This builds the pinned native dependency with the checked-error patch.
./native/build.sh
go run -tags webgpu ./examples/buffers -require-gpu
go run -tags webgpu ./examples/tinygo -require-gpu
WAGO_GPU_TEST=1 go test -race -tags webgpu ./...

# Benchmark all compiled kernels at all four required array sizes.
GOMAXPROCS=1 go run -tags webgpu ./cmd/buffer-bench \
  -kernels=twice,square,copy,polynomial > buffer-results.jsonl

# Matrix, SAXPY, and reduction references. These use authored GPU shaders.
WAGO_GPU_TEST=1 GOMAXPROCS=1 go test -tags webgpu . -run '^$' \
  -bench '^BenchmarkCommon/' -benchtime=3x -count=3 -benchmem
```

`-require-gpu` and `WAGO_GPU_TEST=1` fail when a hardware GPU cannot run.
Software adapters are rejected. Without `webgpu`, or without CGO, the plugin
uses CPU fallback. GPU support is currently enabled only on Linux. Other
platforms use the CPU path.

The GPU dependency has local callback and error-reporting repairs. Read
[the patch notes](third_party/webgpu/PATCHES.md) before changing it. These repairs
include a small native Rust patch and C binding changes. No GPU code or patch is
inside Wago. Native driver calls cannot be forcibly interrupted by a Go context.

## Examples

See the [short example guide](examples/README.md). The hosts now share one small
setup helper, so each program shows its kernel settings and result check.

```sh
go run ./examples/buffers -first-cpu -second-cpu
go run ./examples/tinygo -cpu
go run ./examples/storage -kind f16 -cpu

# WASI guests write Mandelbrot images to stdout.
go run ./examples/mandelbrot > mandelbrot-cpu.pgm
go run ./examples/mandelbrot -program buffers -cpu > mandelbrot-fallback.pgm
go run -tags webgpu ./examples/mandelbrot -program buffers -require-gpu > mandelbrot-gpu.pgm
```

The [Mandelbrot guide](examples/mandelbrot/README.md) explains image settings,
rebuild commands, and the CPU/GPU split. The CPU program is a standalone WASI
command. The buffer program offloads each arithmetic iteration and checks
escape values in the guest. Its readback costs are part of the example.
Images now default to **1920×1080 and 64 iterations**. The example also supports
2560×1440, with a limit of 4,194,304 total pixels.

[The buffer WAT start module](examples/buffers/module.wat) produces `[3, 5, 7, 9]`.
Each dispatch can fall back independently; use `-first-cpu` or `-second-cpu` to
choose a mixed path. The storage example also accepts `-kind memory64` or
`-kind gc` for checked transfer examples.

Rebuild the [complete TinyGo buffer guest](examples/tinygo/guest.go) with:

```sh
tinygo build -target=wasm-unknown -scheduler=none -panic=trap \
  -gc=leaking -opt=2 -no-debug -o examples/tinygo/guest.wasm \
  examples/tinygo/guest.go
```

TinyGo 0.42.0 was tested. `-gc=leaking` is a build setting for these bounded
examples, not a requirement for plugin-owned buffers.

## Host contract

Use `NewHost` for the controlled compilation path:

```go
host, err := wagogpu.NewHost(ctx, wagogpu.Config{
    Kernels: []wagogpu.KernelConfig{{
        ID: 1,
        Export: "wago_gpu.kernel.double",
        CPUExport: "wago_gpu.cpu.double",
        RelaxedFloat: true,
        Bindings: []wagogpu.BindingConfig{
            {Slot: 0, Type: wagogpu.TypeF32, Access: wagogpu.AccessRead},
            {Slot: 1, Type: wagogpu.TypeF32, Access: wagogpu.AccessWrite},
        },
    }},
})
```

Then call `host.Compile(wasm)` and `host.Instantiate(ctx, module)`. Close the
instance, module, and host with an independent cleanup context. The
[example setup helper](examples/internal/runwasm/run.go) shows the lower-level
runtime sequence and returns cleanup errors.

`Host` also offers source preparations with `Compile` and `Close`. It exposes
no artifact-adoption or import-override route. With the lower-level
`PluginSet`, the host must enforce these restrictions itself:

- Use `Runtime.Compile` or `PreparedCompile.Compile`.
- Do not adopt an unrelated compiled artifact. A source digest alone cannot
  prove that an adopted artifact matches inspected source.
- Do not override imports that the shader compiler treats as intrinsics.
- Consume or close prepared compilations. Pending plugin records are bounded;
  an evicted preparation that finishes later is unverified.

Missing exports, wrong signatures, and established contract violations return
`INVALID_KERNEL`. Unsupported GPU instructions use fallback only after current
CPU data is available. Unknown final source metadata is never trusted.

## Guest contract

A kernel exports `(i32 index) -> ()`. Its CPU runner exports `(i32 count) -> ()`.
Names use `wago_gpu.kernel.<name>` and `wago_gpu.cpu.<name>`. The host selects
kernel IDs, names, slot types, access permissions, and float consent.

Buffers belong to one instance. Handles are opaque nonzero `i32` bit patterns.
The guest uses `createBufferPacked`, `bindBuffer`, `setBuffer32/64/GC`,
`dispatch`, `copyBuffer32/64/GC`, and `freeBuffer`. A kernel uses `getBuffer`
and typed `readBuffer*` / `writeBuffer*` intrinsics.

Dispatch status 0 means success. Status 1 means the guest must run its original
CPU runner. Other statuses are errors; do not treat them as fallback.
`BufferSnapshot().Last` preserves the dispatch record through a CPU loop that
uses only scalar intrinsics. Bulk-transfer and management imports update it.

GPU outputs remain private until all checks and completion succeed. A failed
operation does not publish partial output. Device loss or uncertain completion
quarantines the runtime's GPU. Current CPU copies remain usable. GPU-only
contents become lost; a full successful set operation can replace them.

A later trap or cancellation of the enclosing Wasm call does not undo an import
that already committed. Do not automatically retry a chain on that assumption.

## Supported compiler subset

- `local.get`, `local.set`, `local.tee` for i32/f32 values.
- `i32.const`, `f32.const`, `f32.add`, `f32.sub`, `f32.mul`.
- Exact buffer intrinsic imports and a structural final `end`; restricted void
  `return` directly before that `end`.
- One independent invocation per element. Access uses the original index,
  including unchanged copies through locals.

GPU types: I8, U8, I16, U16, I32, U32, F16, F32. Narrow elements use separate
32-bit GPU words. I64, U64, and F64 have CPU imports and require fallback.
There is no general integer arithmetic, gather/scatter, reduction, matrix
compiler, arbitrary control flow, or automatic function replacement.

`CompileBufferWGSL` returns inspectable WGSL. Its `*CompileError` category is
available through `errors.As`, including after wrapping. Export names are not
inserted into shader identifiers.

F16 conversion and float copies use integer bits. Scalar F16 conversions use
round-to-nearest, ties-to-even and canonical NaNs. Bulk copies preserve bits.
GPU float arithmetic can fuse operations and change subnormal or non-finite
behavior. `RelaxedFloat` is required per float kernel. Exact transfer is not a
promise of exact arithmetic.

## Limits, measurements, and maintenance

Defaults are 64 buffers per instance, 64 selected kernels, 8 slots per kernel,
10 million elements per buffer, 256 MiB tracked instance storage, and 512 MiB
tracked runtime storage. Hosts can lower them. Each instance can retain up to
eight output scratch buffers, one readback buffer, one uniform buffer, and one
bounded Go conversion slice. Idle capacity is reclaimed before rejecting a
budget reservation. Driver overhead is outside these byte counts.

Snapshots distinguish submission from confirmed execution, guest copies from
host/device transfers, and device-local seed copies. GPU timestamps are not
available: `TimestampsValid` is false. Profile fields are host wall times and
include queue waits. `ProfileStages` adds waits, so benchmark results state
whether it is enabled.

Source-transform hooks disable Wago prepared-artifact cache eligibility.
Pipelines are cached while their modules or instances own them. No Wago public
API change is required under the documented host restrictions.

```sh
go generate ./...                 # Requires WABT and wasm-tools for fixtures.
python3 spec/check_spec.py --tinygo
go test -run '^$' -fuzz FuzzBufferCompiler -fuzztime 10s .
```

The module path and plugin provenance identify `jtenner/wago-gpu`. The ABI
remains provisional in `v0.0.0`. Historical
scalar measurements remain in [REPORT.md](REPORT.md) and
[IMPROVEMENTS.md](IMPROVEMENTS.md); they are not buffer benchmark results.
