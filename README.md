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

# Benchmark both math kernels at all four required array sizes.
go run -tags webgpu ./cmd/buffer-bench > buffer-results.jsonl
```

`-require-gpu` and `WAGO_GPU_TEST=1` fail when a hardware GPU cannot run.
Software adapters are rejected. Without `webgpu`, or without CGO, the plugin
uses CPU fallback. GPU support is currently enabled only on Linux. Other
platforms use the CPU path.

The GPU dependency has local callback and error-reporting repairs. Read
[the patch notes](third_party/webgpu/PATCHES.md) before changing it. These repairs
include a small native Rust patch and C binding changes. No GPU code or patch is
inside Wago. Native driver calls cannot be forcibly interrupted by a Go context.

## Complete examples

[The WAT start module](examples/buffers/module.wat) creates two F32 buffers,
computes `B = 2*A`, then `A = B+1`, and copies `[3, 5, 7, 9]` to guest memory.
Its [Go host](examples/buffers/main.go) checks the result after instantiation.
Each dispatch can fall back independently:

```sh
go run -tags webgpu ./examples/buffers -first-cpu
go run -tags webgpu ./examples/buffers -second-cpu
```

[The complete TinyGo guest](examples/tinygo/guest.go) uses the single-result
packed allocation import. It includes allocation, transfer, dispatch, CPU
fallback, result checks, and cleanup. Rebuild its checked-in Wasm with:

```sh
tinygo build -target=wasm-unknown -scheduler=none -panic=trap \
  -gc=leaking -opt=2 -no-debug -o examples/tinygo/guest.wasm \
  examples/tinygo/guest.go
```

TinyGo 0.42.0 was tested. The tiny guest uses fixed arrays. `-gc=leaking` is a
build setting for this example, not a requirement for plugin-owned buffers.

Typed storage examples are also available:

```sh
go run ./examples/storage -kind f16 -cpu
go run ./examples/storage -kind memory64 -cpu
go run ./examples/storage -kind gc -cpu
go run -tags webgpu ./examples/storage -kind f16 -require-gpu
```

The [storage WAT files](examples/storage/main.go) show F16 upcasting and explicit
transfers. The memory64 and GC examples test transfers; they do not dispatch GPU
work. Regeneration needs `wasm-tools`.

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
instance, module, and host with an independent cleanup context. See the TinyGo
host for a complete sequence that returns cleanup errors.

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
`BufferSnapshot().Last` preserves the dispatch record through the CPU loop.
Scalar intrinsics do not replace it.

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
tracked runtime storage. Hosts can lower them. One idle scratch buffer per
instance can be reused; idle capacity is reclaimed before rejecting a budget
reservation. Driver overhead is outside these byte counts.

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
