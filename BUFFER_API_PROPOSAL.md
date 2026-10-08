# Wago GPU plugin specification

> Implementation update (2026-10-07): the buffer contract below now has an
> implementation, CPU tests, fake-backend tests, and hardware evidence. The ABI
> remains provisional for publication review. “Proposed” in retained design text
> identifies the provisional contract, not current implementation absence. See
> [BUFFER_REPORT.md](BUFFER_REPORT.md) for current support and limits. Sections
> that describe the original binding or earlier runs are historical evidence.

Status: Revised draft; `wago_gpu_v1` is not frozen. Date: 2026-10-07.

This revision addresses both boundary reviews in section 31. Confirmed dependency defects and untested requirements are release gates. Historical arithmetic runs do not prove that those gates have passed.

This document specifies the complete `wago-gpu` plugin contract. It includes the existing scalar interface and the proposed buffer interface. It retains its original filename so existing links continue to work. Implementation evidence is reported separately in BUFFER_REPORT.md.

`MUST` identifies a requirement. `SHOULD` identifies a preferred implementation. `MAY` identifies an allowed choice. Sections marked **Existing** describe the inspected code. Sections marked **Proposed**, and the buffer requirements in sections 1–17, define the intended extension. Where the code and a proposed requirement differ, the proposal is a development requirement, not a claim about current behavior.

The current implementation is described in [README.md](README.md) and [BUFFER_REPORT.md](BUFFER_REPORT.md). [REPORT.md](REPORT.md) and [IMPROVEMENTS.md](IMPROVEMENTS.md) retain the earlier scalar measurements.

## Document map

| Sections | Subject |
| --- | --- |
| 1–4 | Scope, terms, named kernels, buffer ownership |
| 5–7 | Version 1 imports, element types, compiler rules |
| 8–12 | Data consistency, failure, status, arithmetic, resource limits |
| 13–14 | Complete start example and Wago integration |
| 15–17 | Tests, measurements, implementation order |
| 18 | Complete existing scalar ABI and compiler contract |
| 19–20 | Existing and proposed Go host API and configuration |
| 21–22 | Registration, authorities, compilation, and lifetime |
| 23–24 | GPU backend, shader layout, and operation semantics |
| 25 | Diagnostics and measurement fields |
| 26–27 | Builds, examples, compatibility, and non-goals |
| 28–30 | Acceptance checklist, existing evidence, and specification checks |
| 31 | Review decisions, evidence, and release gates |

## 1. Purpose and scope

The plugin lets a Wasm module use named compute kernels and typed buffers. The same kernel body can run on the GPU or through Wago's native CPU execution. A buffer can retain its data on the GPU between kernel calls.

The plugin MUST remain an optional Go project outside Wago. It MUST own GPU devices, shaders, pipelines, buffers, transfers, and cleanup. It MUST use Wago's public plugin and checked guest-storage APIs. It MUST NOT add GPU execution to Wago's native compiler or runtime.

Version 1 accelerates independent elementwise operations. Reductions, matrix products, scans, and stencils are outside its GPU subset. Version 1 uses explicit GPU calls. It does not replace ordinary function calls automatically. It does not include a general Wasm compiler, full SSA, a large optimizer, textures, arbitrary pointers in shaders, or automatic recovery through kernel replay.

### Scalar and buffer interfaces

| Item | Legacy scalar interface | Buffer interface |
| --- | --- | --- |
| Kernel selection | One selected export per module | Explicit table of named kernel exports |
| Kernel signature | `(f32) -> f32` | `(index: i32) -> ()` for buffer kernels |
| GPU calls | `wago_gpu.run` and `run_batch` | Versioned buffer and dispatch imports |
| Data | Input and output in memory 0 | Plugin-owned buffers and explicit transfers |
| Data between passes | Same kernel, retained during a batch | Different kernels, retained between calls |
| Element types | `f32` | Typed integer and float access |
| Guest storage on the GPU path | Unshared wasm32 memory | Checked wasm32, wasm64, and numeric GC-array transfers |
| Buffer hardware evidence | Not applicable | Recorded in BUFFER_REPORT.md; release limits remain |

The existing imports, signatures, and status codes MUST remain unchanged. The new imports use module name `wago_gpu_v1`. An import module version identifies its contract; it is not a Wago version.

## 2. Terms

| Term | Meaning |
| --- | --- |
| Buffer | A plugin-owned, typed array with a fixed element count |
| Handle | An opaque `i32` value that identifies a buffer in one instance |
| Slot | A small numbered binding used by a kernel to select a buffer |
| Kernel | An exported Wasm function selected for GPU compilation |
| Dispatch | One request to run a kernel for indices `0` through `count - 1` |
| CPU copy | Buffer contents in memory owned by the Go plugin |
| GPU copy | Buffer contents in device storage |
| Current copy | A copy that contains the latest successfully committed contents |

A handle is not a Wasm pointer, a Go pointer, a GC reference, or a GPU address. A slot is not a handle. A wasm64 address does not make a handle or element index 64 bits wide.

## 3. Named kernel contract

A buffer kernel MUST have an explicit export name:

```text
wago_gpu.kernel.<name>   (param index i32)
wago_gpu.cpu.<name>      (param count i32)
```

The first export performs the work for one element. The second export is a CPU runner. It loops through the requested indices and calls the same kernel body as an ordinary Wasm function. Wago executes that body through its native CPU path. The plugin MUST NOT substitute a Go interpreter for this baseline.

The host MUST supply a kernel table. Each entry specifies:

- A nonzero `i32` kernel ID.
- The exact kernel export name and CPU runner export name.
- The required slots, element types, and access modes: `READ`, `WRITE`, or `READ_WRITE`.
- Whether relaxed floating-point execution is allowed.

For example, the complete module in section 13 uses this table:

| ID | Kernel export | CPU runner export | Slot 0 | Slot 1 |
| --- | --- | --- | --- | --- |
| 1 | `wago_gpu.kernel.double` | `wago_gpu.cpu.double` | `F32`, `READ` | `F32`, `WRITE` |
| 2 | `wago_gpu.kernel.addOne` | `wago_gpu.cpu.addOne` | `F32`, `READ` | `F32`, `WRITE` |

This table describes host configuration. It is not an existing Go configuration type.

The plugin MUST check export names and signatures. Duplicate IDs or names are invalid configuration. Kernel IDs MUST NOT depend on Wasm function indices or export order. Only host-selected exports can become GPU kernels. A name prefix alone does not authorize compilation.

A WAT label such as `$double` is internal to the module. The exported string is the public contract. Existing scalar kernels keep their existing configuration and signature.

The guest is responsible for calling the correct CPU runner when dispatch returns fallback. The host configuration and guest MUST agree on IDs. The plugin checks runner signatures; it does not prove that an arbitrary CPU loop has the intended behavior.

## 4. Buffer ownership and lifetime

`createBuffer` creates logical storage on the CPU. It MUST work without a GPU. The element count and type are fixed until the buffer is freed. All initial elements have zero bits. GPU allocation can occur later, when a dispatch needs it.

Handles belong to one Wasm instance within one plugin runtime. Handle `0` is invalid. All other bit patterns, including values with the sign bit set, are opaque tokens. Token exhaustion returns `LIMIT_EXCEEDED`. The plugin MUST reject handles from another instance in that runtime, freed handles, and unknown handles. It MUST prevent a stale handle from selecting a later allocation. A runtime-wide token allocator with an owner check is sufficient. An implementation can use generation numbers, but it MUST report exhaustion instead of wrapping into a valid stale handle. Handles have no meaning in a different runtime or process and cannot be persisted for later use.

`freeBuffer` invalidates a handle and clears its slot bindings. Native GPU resources MUST remain alive until their submitted work has finished. Instance cleanup MUST release buffers that the guest did not free. Module cleanup MUST retain resources needed by live instances. Runtime cleanup MUST finish device cleanup.

Version 1 has no buffer resize, shared handles across instances, or automatic link to guest memory. A transfer copies data. Later guest-memory changes do not change a buffer until another transfer occurs.

## 5. Import interface

All imports in this section belong to `wago_gpu_v1`.

An `i32` count, element offset, slot, memory index, or kernel ID is interpreted as unsigned. An `i64` guest address is interpreted as unsigned. Large values MUST fail bounds or limit checks; the plugin MUST NOT interpret them as negative offsets. Status values use the table in section 10.

### 5.1 Management and execution

| Import | Wasm signature | Purpose |
| --- | --- | --- |
| `createBuffer` | `(elementType: i32, count: i32) -> (handle: i32, status: i32)` | Allocate zero-filled CPU storage |
| `createBufferPacked` | `(elementType: i32, count: i32) -> packed: i64` | Single-result form of `createBuffer` |
| `freeBuffer` | `(handle: i32) -> status: i32` | Release a buffer |
| `bindBuffer` | `(slot: i32, handle: i32) -> status: i32` | Set a slot; handle `0` clears that slot |
| `getBuffer` | `(slot: i32) -> handle: i32` | Select the buffer bound to a slot |
| `dispatch` | `(kernelID: i32, count: i32) -> status: i32` | Try GPU execution; report success or CPU fallback |

`createBuffer` returns `(0, errorStatus)` on failure. Its result order is significant: the status is on top of the Wasm operand stack.

`createBuffer`, `createBufferPacked`, `freeBuffer`, `bindBuffer`, and `dispatch` are CPU-side control calls. They MUST NOT occur inside a GPU kernel. `getBuffer` is valid on both execution paths. It selects a binding; it does not copy data back to the guest.

`dispatch` is synchronous. Success means the work has completed and its output buffer versions have been committed. It does not imply that results have been copied to guest memory. The guest requests that copy separately.

Before execution, dispatch MUST validate the kernel ID, required bindings, types, access declarations, and lengths. Each required buffer MUST contain at least `count` elements. A handle MUST NOT be bound to more than one declared slot of the selected kernel, including unused declarations. Bindings in slots not declared by that kernel are excluded from this check. Temporary duplicate bindings between calls are allowed. An in-place kernel uses one `READ_WRITE` slot.

A zero-count dispatch still validates its contract and limits. If it reaches the import, it then returns success before checking cancellation or obtaining contents, without GPU work. Diagnostics MUST identify this case as a no-op, not a hardware execution.

### 5.1.1 Packed result and canonical ABI

`createBufferPacked` has the same allocation and error behavior as `createBuffer`. Its unsigned 64-bit result is `(uint64(uint32(status)) << 32) | uint64(uint32(handle))`. Extract the low 32 bits as the handle and the high 32 bits as the status. Failure has a zero handle. A wrapper MUST NOT sign-extend the handle or discard a nonzero status.

The draft ABI now contains **34 imports**: six management/access-selection imports, six bulk transfers, and 22 scalar access imports. The packed companion adds one import to the earlier 33-import proposal. This is a pre-freeze change. It does not change the two-result import used by the WAT example.

[spec/abi_v1.json](spec/abi_v1.json) is the canonical table for module name, signatures, type IDs, status values, and named Go constants. Implementation MUST generate guest declarations, registration descriptors, and signature tests from that table, or mechanically compare them against it. The file describes a proposed ABI; it does not register any imports in the current plugin. Every declared CPU import is required before a build claims complete v1 support.

### 5.2 Transfers from and to guest storage

The following operations copy a range of elements. `bufferOffset` and `count` are in elements. `guestByteOffset` is in bytes. `arrayOffset` is in GC-array elements.

```text
setBuffer32(handle: i32, bufferOffset: i32,
            memoryIndex: i32, guestByteOffset: i32, count: i32) -> status: i32

setBuffer64(handle: i32, bufferOffset: i32,
            memoryIndex: i32, guestByteOffset: i64, count: i32) -> status: i32

setBufferGC(handle: i32, bufferOffset: i32,
            array: anyref, arrayOffset: i32, count: i32) -> status: i32

copyBuffer32(handle: i32, bufferOffset: i32,
             memoryIndex: i32, guestByteOffset: i32, count: i32) -> status: i32

copyBuffer64(handle: i32, bufferOffset: i32,
             memoryIndex: i32, guestByteOffset: i64, count: i32) -> status: i32

copyBufferGC(handle: i32, bufferOffset: i32,
             array: anyref, arrayOffset: i32, count: i32) -> status: i32
```

`setBuffer*` copies guest data into a buffer. `copyBuffer*` copies buffer data into existing guest storage. Neither operation allocates guest memory or a guest GC array. These calls are CPU-side calls only.

The `32` and `64` variants MUST match the selected memory's address type. Version 1 rejects shared memories. A memory index selects the actual guest memory; it does not select a GPU binding.

Both source and destination ranges MUST be checked before mutation. Size multiplication and end-offset calculations MUST detect overflow. Guest memory uses little-endian storage. Unaligned guest addresses are allowed if Wago's checked byte-range API permits them.

Bulk transfers preserve storage bits. They do not convert between element types. A partial `setBuffer*` MUST preserve elements outside its range. A failed transfer MUST leave the destination unchanged. A zero-count transfer may use an offset at the end of storage, but its handle, storage kind, and offsets still require validation.

The plugin MUST obtain a current CPU copy before a partial CPU update or a guest copy-back. It can skip that readback for an update that replaces the complete buffer. A complete replacement can restore a buffer whose old contents were lost.

For GC transfers, `anyref` is checked dynamically. It MUST refer to a numeric array with the storage kind in section 6. Null references, structs, and reference-element arrays are invalid. An immutable numeric array can be a source. A destination array MUST be mutable.

The implementation MUST use `WithGuestStorage`, `MemoryInfo`, `MemoryRange`, and the GC-array APIs. Borrowed slices and `GuestGCRef` values MUST NOT survive their callback. CPU-owned transfer data can survive it. The plugin MUST NOT retain a guest pointer as buffer storage or call back into the guest while a storage view is borrowed.

Before a GC byte view is requested, call `GCArrayInfo`, check the requested range, and calculate its storage cost with checked arithmetic. In the pinned Wago implementation, reading an immutable numeric array through `GCArrayBytes` copies the **whole payload**, even for a small requested range. Reserve that full temporary cost before the call. Return `LIMIT_EXCEEDED` if it cannot fit. Complete a valid zero-count transfer before requesting any byte view. A future checked GC-array range-copy API could remove that extra cost; it is not available in the pin.

### 5.3 Scalar access on the CPU and GPU

For each type suffix `T` in section 6, the interface provides:

```text
readBufferT(handle: i32, index: i32) -> value: WasmType(T)
writeBufferT(handle: i32, index: i32, value: WasmType(T)) -> ()
```

For example:

```text
readBufferF16(i32, i32) -> f32
readBufferF32(i32, i32) -> f32
readBufferI16(i32, i32) -> i32
readBufferU16(i32, i32) -> i32
writeBufferF16(i32, i32, f32) -> ()
writeBufferF32(i32, i32, f32) -> ()
```

On the CPU, these imports access the plugin's current CPU copy. A first read after GPU execution can require a readback. Subsequent reads of that version MUST reuse the copy.

On the GPU, the compiler translates these calls into shader buffer accesses. There is no Go call, Wasm host call, or host readback for each GPU element. `getBuffer` is also a compiler intrinsic on this path.

A scalar access MUST match the buffer's exact type. `readBufferI32` cannot read an `F32` buffer. Invalid scalar access or an unbound `getBuffer` MUST trap through Wago's host-trap mechanism. It MUST NOT return a default value. Control and bulk-transfer imports return status values instead of trapping for ordinary contract errors.

## 6. Element types and storage

| ID | Suffix | Stored bytes per element | Wasm scalar type | CPU read behavior | GC storage kind | GPU support plan |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `I8` | 1 | `i32` | Sign extend | `i8` | After F32 phase |
| 2 | `U8` | 1 | `i32` | Zero extend | `i8` | After F32 phase |
| 3 | `I16` | 2 | `i32` | Sign extend | `i16` | After F32 phase |
| 4 | `U16` | 2 | `i32` | Zero extend | `i16` | After F32 phase |
| 5 | `I32` | 4 | `i32` | Preserve bits | `i32` | After F32 phase |
| 6 | `U32` | 4 | `i32` | Preserve bits | `i32` | After F32 phase |
| 7 | `F16` | 2 | `f32` | Convert binary16 to f32 | `i16` containing float bits | After F32 phase |
| 8 | `F32` | 4 | `f32` | Preserve bits | `f32` | First phase |
| 9 | `I64` | 8 | `i64` | Preserve bits | `i64` | CPU only in version 1 |
| 10 | `U64` | 8 | `i64` | Preserve bits | `i64` | CPU only in version 1 |
| 11 | `F64` | 8 | `f64` | Preserve bits | `f64` | CPU only in version 1 |

Each row defines separate `readBuffer` and `writeBuffer` import names. Wasm integer result types do not encode signedness; the import name defines extension behavior. Integer writes retain the low stored-width bits, as a narrow Wasm store does.

“Stored bytes” defines the CPU and guest-transfer format. It does not require an identical GPU layout. An F16 buffer stores two bytes per element in guest storage, but its initial GPU representation can use four bytes per element.

For the first GPU implementation, each element of width 32 bits or less MUST occupy a separate 32-bit storage word. Narrow values use the low bits; unused high bits are zero. This avoids two invocations updating different narrow elements in the same word. Tightly packed writable GPU storage is deferred until the implementation can prove exclusive word ownership or provide a correct atomic update.

The portable GPU path MUST NOT silently narrow 64-bit types. Kernels that require those types use CPU fallback. WGSL provides concrete 32-bit scalar types and an optional `f16` extension; this proposal does not assume native 8-bit, 16-bit integer, or 64-bit scalar storage. See the [WGSL scalar type rules](https://www.w3.org/TR/WGSL/#scalar-types).

### 6.1 Half-float conversion

`F16` means IEEE 754 binary16 storage. `readBufferF16` MUST return an `f32` on both the CPU and GPU paths. Finite binary16 values widen exactly. The import does not expose a Wasm `f16` value.

The conversion contract is:

- Preserve the sign of zero and infinity.
- Convert every binary16 NaN to the quiet positive f32 NaN with bits `0x7fc00000`.
- `writeBufferF16` accepts an `f32` and rounds to binary16 with round-to-nearest, ties-to-even.
- Preserve representable half-float subnormal values. Underflow produces the correctly signed zero when rounding requires it. Overflow produces the correctly signed infinity.
- Store every NaN as the binary16 quiet positive NaN `0x7e00`.

Bulk `setBuffer*` and `copyBuffer*` operations preserve raw bits, including NaN payloads. The canonical NaN rules apply to numeric F16 reads and writes.

Version 1 performs arithmetic on the widened `f32` value. It rounds to F16 only when `writeBufferF16` is called. It MUST NOT silently introduce half-precision arithmetic between loads and stores.

Exact storage and conversion values MUST remain integer-encoded bits inside the shader compiler. F16 widening produces an f32 bit record. Copies, local assignments, and conversion-only stores retain that record without materializing a WGSL floating-point value. F32 storage MUST also have an integer-bit access path for these operations. Materialize a WGSL f32 only when arithmetic needs it; that operation enters the relaxed numerical contract. An arithmetic result can be encoded again, but this does not restore information that relaxed arithmetic discarded.

Do not emit a NaN-valued or infinity-valued WGSL constant expression, including in an unused helper branch. Such constants can cause shader creation to fail. The exact conversion helpers MUST use integer operations and integer constants. Finite binary16 subnormals widen to normal f32 values, so they do not become f32 subnormals. See the [WGSL float evaluation rules](https://www.w3.org/TR/WGSL/#floating-point-evaluation).

A backend that cannot preserve the conversion-only contract MUST use CPU fallback for that kernel. Native f16 conversion is optional and requires the same tests. Test F16-to-F32 storage bits without intervening arithmetic, and `writeBufferF16(readBufferF16(x))` for all 65,536 stored patterns, with the stated NaN canonicalization. These tests are separate from arithmetic-tolerance tests.

## 7. GPU compiler rules

The compiler MUST translate the selected function's actual Wasm instructions. It MUST NOT choose a hardcoded shader from a function name or example ID.

The initial buffer-kernel subset adds these operations to the current small arithmetic compiler:

- `local.get`, `local.set`, and `local.tee` for the index, values, and tracked buffer selections.
- `i32.const` for constant slot selection.
- `f32.const`, `f32.add`, `f32.sub`, and `f32.mul`.
- Calls to the exact versioned `getBuffer`, `readBufferF32`, and `writeBufferF32` imports.
- Function termination through the final `end`, or one void `return` immediately followed by that structural final `end`. No other trailing instruction bytes are accepted.

The typed phase adds the corresponding narrow integer and F16 access intrinsics. It does not imply support for every integer arithmetic instruction. Each new instruction requires an explicit lowering and tests.

In the first version:

- A `getBuffer` slot MUST be a compile-time constant in a GPU kernel.
- A buffer handle can pass through tracked locals into access intrinsics. It cannot be used as an address, stored to guest memory, or used in arithmetic.
- Each scalar buffer access MUST use the kernel's unmodified index. Neighbor reads, gather, scatter, and cross-invocation communication are outside this subset.
- Reads and writes MUST agree with the selected kernel's slot access modes.
- The generated entry point MUST check the dispatch count before accessing storage. Extra invocations in the last workgroup MUST return without work.
- Guest memory operations, GC operations, allocation, free, rebinding, dispatch, arbitrary calls, and loops inside a GPU kernel are unsupported.

These restrictions let the compiler prove bounds and avoid data races without a large optimizer. CPU runner loops remain ordinary Wasm and are not compiled into shaders.

Unsupported instructions or types reject GPU compilation for that kernel. A known contract violation is classified separately, as section 7.3 specifies. Neither case invalidates an otherwise valid CPU-only Wasm module. Wago validates the module. The plugin independently decodes the selected code with bounded work and rejects any form it cannot inspect safely.

Support for GC types and memory64 in the surrounding module requires changes to the plugin's current module decoder. A checked Wago transfer API alone does not make the current GPU decoder support those modules.

Pipeline cache keys MUST include source identity, the selected export, slot types and access, compiler version, numerical mode, and relevant device features. A later source transform that changes the inspected module MUST invalidate the generated shader. Pipeline creation failure selects CPU fallback when current CPU data is available.

### 7.1 Value records, locals, and limits

Each decoded operand is the value produced at that instruction. Later local assignments or buffer writes MUST NOT change an earlier stack value. Keep a bounded table of computed values. Stack entries refer to immutable records; locals refer to their current records. Emit a saved read before a later write. Do not delay evaluation of a saved local or buffer load.

For example, with local x equal to 1, `(f32.add (local.get x) (local.tee x (f32.const 2)))` produces 3. Replacing both operands with a later read of x would be incorrect. Reusing a record also prevents repeated expansion of a large expression.

Track these value categories through local copies:

| Category | Allowed use |
| --- | --- |
| Known i32 constant | Constant slot selection, or i32 data in a supported typed store |
| Original invocation index | Element index or i32 data in a supported typed store, including copies through locals |
| Selected buffer slot | Handle operand to a matching intrinsic |
| Ordinary numeric value | Supported arithmetic or typed store |
| Exact float-bit value | Copies and exact conversion/storage; materialize only for arithmetic |

The original index is a value, not a local number. A saved copy remains valid after parameter 0 is overwritten. A local has the original-index category exactly when its current value record is that original value. Assigning another value removes the category; assigning a saved original-index value restores it. Numeric zero initialization of an i32 local does not create a valid handle or an original-index value. Reassigning a handle local updates only its current category. GPU buffer kernels accept only i32 and f32 additional locals in v1; their Wasm zero initialization MUST be preserved.

Known i32 constants and the original invocation index can also be the value operand of I8/U8/I16/U16/I32/U32 stores. Preserve their 32-bit value, subject to narrow-write truncation. Thus constant fill and index fill are supported without integer arithmetic. Local copies keep this permission. A tracked buffer handle is not an integer data value and MUST be rejected as a store value.

Materializing an exact float-bit record for one arithmetic use MUST NOT change that record. Another use can still copy its original bits exactly. Test one F32 load feeding both relaxed arithmetic and an exact copy output.

Before expanding any declaration or allocating decoder state, check these fixed limits with overflow-safe addition: 4,096 locals including parameters, 4,096 stack entries, 8,192 value records, and 1 MiB of accounted temporary expression/value storage per kernel. A compact declaration with a huge repetition count MUST fail before allocation. Source and generated-WGSL limits still apply. Both the source hook and `CompileBufferWGSL` enforce these limits.

`i32.const` uses signed LEB128, at most five bytes with valid sign extension in the last byte. Accept valid signed encodings, including legal non-minimal encodings; reject truncation, overflow, and an unterminated fifth byte. Record the resulting 32-bit pattern, then apply unsigned interpretation only where the ABI requires it.

Generate private shader identifiers from compiler-assigned numbers. Export names MUST NOT be inserted into WGSL identifiers or unescaped comments.

### 7.2 Exact GPU instruction set by phase

| Operation | F32 GPU phase | Narrow-type GPU phase |
| --- | --- | --- |
| `local.get`, `local.set`, `local.tee` | i32/f32 locals with tracked values | Same |
| `i32.const` | Tracked constants; no integer arithmetic implied | Same |
| `f32.const`, `f32.add`, `f32.sub`, `f32.mul` | Supported under the float policy | Same |
| Direct `call` | Exact `getBuffer`, `readBufferF32`, `writeBufferF32` intrinsics only | Also I8/U8/I16/U16/I32/U32/F16 read/write intrinsics |
| `return`, final `end` | Only the termination form stated above | Same |
| All other guest opcodes | GPU rejection | GPU rejection |

In particular, integer-buffer access does not add `i32.add`, multiplication, shifts, comparisons, conversions, or reinterpret instructions to guest GPU code. Generated integer conversion helpers may use WGSL integer operations internally. CPU code can use any instruction Wago supports. I64/U64/F64 access remains CPU-only in complete v1.

### 7.3 Translation result categories

| Condition | Result |
| --- | --- |
| Invalid host configuration | `New` returns an error |
| Missing export, wrong signature, or unverified final contract | `INVALID_KERNEL` on dispatch |
| Proven write to READ, use of an undeclared slot, or wrong typed intrinsic | `INVALID_KERNEL` on dispatch |
| Validated contract, valid bindings, unsupported GPU instruction or capability | `CPU_FALLBACK` only after current CPU data is ready |
| Wrong buffer type or missing/aliased binding at dispatch | The corresponding type or binding status |
| Lost current contents | `CONTENTS_LOST`; never stale-data fallback |

Use an internal typed error category, not error-message matching. Metadata inspection is separate from GPU lowering. Lowering need not prove arbitrary unsupported CPU code, but a contract violation it has established MUST NOT be relabeled as unsupported translation. A decoder failure that prevents trustworthy metadata validation leaves the contract unverified, rather than CPU-fallback-ready.

An established contract violation belongs to the CPU contract record, including while that record is pending attachment. Preserve it through WGSL eviction, pipeline failure, and GPU-disabled operation. Only unsupported GPU capability/lowering details can be discarded independently. Contract checks for the supported body forms run independently of device availability and the Disabled setting. This does not require proving arbitrary unsupported CPU code. If the entire unattached contract record is evicted under section 22.1.1, its later completion is unverified and still returns `INVALID_KERNEL`; it never becomes valid fallback.

### 7.4 Actual read and write sets

For every successfully lowered body, record the slots actually read and actually written by accepted intrinsics. These sets are separate from the declared permissions. The straight-line v1 subset requires only bounded sets of slot numbers, not general control-flow analysis.

Validate every declared slot at dispatch, including unused slots. Only slots in the actual write set are outputs: they receive temporary storage and a new committed version after a nonzero successful dispatch. A READ_WRITE slot used only for reads retains its version. An unused WRITE declaration does not allocate an output, seed it, or invalidate a CPU copy. A real store creates a new version even when it writes the same bits. A zero-count call creates no versions.

Slots outside the actual write set MUST NOT have current copies invalidated by dispatch. A required upload or readback can populate an additional current copy without changing the content version. Unused declared slots require contract validation but no data transfer. If GPU lowering is unavailable, do not invent actual-use sets for arbitrary CPU code; fallback preparation retains its conservative contract in section 9.

## 8. CPU and GPU data consistency

Each buffer has a committed content version and one of these states:

| State | Meaning |
| --- | --- |
| `CPU_CURRENT` | Only the CPU copy is current |
| `GPU_CURRENT` | Only the GPU copy is current |
| `BOTH_CURRENT` | Both copies have the same current version |
| `CONTENTS_LOST` | No usable copy of the latest committed version remains |
| `RETIRED` | The handle is invalid; resource cleanup can still be pending |

The required state changes are:

1. Create: zero-filled `CPU_CURRENT`.
2. CPU write or guest upload: new `CPU_CURRENT` version.
3. Upload a current CPU version to the device: `BOTH_CURRENT`.
4. Successful GPU write: new `GPU_CURRENT` version.
5. Successful readback of that version: `BOTH_CURRENT`.
6. Free: `RETIRED`.

Read-only GPU use, determined by the actual write set rather than the access declaration, does not create a new version. Intermediate GPU results MUST remain usable by the next dispatch without a CPU transfer. GPU command order and storage barriers MUST make the preceding writes visible.

The CPU copy is a cache. It MUST NOT be treated as current merely because it exists. Version checks apply to scalar reads, partial writes, copy-back, and fallback.

The initial implementation serializes GPU submissions and calls that change instance bindings. The host MUST NOT call the same instance concurrently while a CPU runner or binding sequence is active. Different instances can run concurrently at the Wago level; their plugin resources remain isolated and GPU work can be serialized.

## 9. Failure and fallback contract

`dispatch` returns `CPU_FALLBACK` only when the guest can safely run the original CPU kernel using the buffer contents from the start of that call. Before returning that status, the plugin MUST make all required bound buffers current on the CPU. This can require a readback from an earlier successful dispatch.

No GPU output from a failed dispatch may replace the previous committed buffer version. The implementation MUST preserve that version until completion. It can use reusable temporary output buffers and swap them into place after success. A `READ_WRITE` output needs the old contents before execution. Elements outside the dispatch range MUST retain their old values.

No failed GPU operation may copy partial results into guest memory. The guest makes a separate checked copy-back call after successful GPU or CPU execution.

### 9.1 Missing device and compile rejection

No GPU, an unsupported kernel, unavailable device features, a configured CPU threshold, or shader creation failure normally causes `CPU_FALLBACK`. Buffer creation, guest transfers, scalar access, and native CPU execution remain available.

Malformed import signatures, invalid handles, invalid ranges, and missing bindings are contract errors. They MUST NOT be disguised as fallback.

### 9.2 Device loss and stale copies

If a device loses the only current copy of a buffer, its state becomes `CONTENTS_LOST`. The plugin MUST report that status. It MUST NOT run the CPU kernel on an older CPU copy and call that a successful fallback.

A buffer with a current CPU copy remains recoverable after device loss. The guaranteed repair for a lost buffer is a successful full-range `setBuffer32`, `setBuffer64`, or `setBufferGC`, with buffer offset zero and count equal to its length, or a new buffer. No read, no-op, partial update, or dispatch repairs lost contents in v1. Old in-flight storage remains retained and charged after a CPU replacement.

This is a limit of retaining data only on the GPU. Automatic replay, a CPU checkpoint after every dispatch, and atomic rollback of a complete multi-kernel chain are outside version 1. A caller that needs recovery across device loss must retain enough input to repeat the chain.

### 9.3 Cancellation and CPU errors

For a nonzero operation, cancellation before submission returns `CANCELLED` without changes. A validated zero-count import follows the no-op precedence in section 24.1. If a timeout leaves completion uncertain, the plugin MUST stop using that device for new work and retain in-flight resources until cleanup is safe. It MUST distinguish preserved contents from lost contents. A timeout setting does not guarantee that a blocked native driver call can be interrupted.

CPU fallback is an ordinary Wasm loop. A trap in that loop can leave partial CPU-buffer writes, as an ordinary Wasm loop can leave partial memory writes. The proposal does not promise a transaction around arbitrary CPU execution. Bulk guest copy-back remains explicit and MUST validate its full destination before mutation.

Each dispatch commits separately. A later failure does not undo earlier successful kernels in a chain.

### 9.4 One terminal commit decision

Each dispatch has one commit decision under the instance lock. Cancellation observed before that decision prevents publication. Once the decision authorizes a commit, publish all outputs and report `OK`; later cancellation cannot turn that result into failure. A dispatch that reports `CANCELLED` has not published GPU output.

Each dispatch has one terminal result. A late callback can release its own resources, but cannot change that result, publish outputs, replace a newer CPU version, restore a freed handle, or reopen an instance. Associate each submission with its instance generation, source buffer versions, and a non-reused operation token. Callback delivery and result publication are separate actions. Allocating staging or populating an unchanged mirror is not a content commit.

These terminal-result rules apply to the plugin import, not to the enclosing Wago invocation. That invocation can report cancellation or trap after an import commits, and the guest might not save the import status. An outer failure does not roll back committed buffer versions or completed guest copies. The host MUST NOT infer unchanged data or automatically repeat a chain from an outer cancellation error alone. Recovery or retry needs an application-defined checkpoint or knowledge of completed work.

Pinned Wago checks the context again after native execution returns; see [contextInterruptError](https://github.com/wago-org/wago/blob/7aa401f29a33eb9050559aab5488cdb307f5a078/src/wago/instance_call.go#L135). Test this with a real Wago invocation and a fake backend: cancel after the commit decision and verify an outer cancellation can coexist with committed contents. Also test a guest trap after successful dispatch and copy-back.

### 9.5 Device health, quarantine, and both ABIs

| Device state | Permitted behavior |
| --- | --- |
| `USABLE` | GPU execution and checked readback |
| `QUARANTINED` | No new submissions, uploads, or readbacks; retain uncertain native resources |
| `LOST` or `ABANDONED` | CPU operations on current copies; GPU-only data is unavailable |
| `CLOSING` | Reject new work; perform independent safe cleanup |

Version 1 cannot leave quarantine. A new plugin runtime is required to retry GPU execution. Existing callback polling is permitted only for retirement and error resolution, not to recover quarantined outputs or data. Because GPU data cannot be read in quarantine, its logical consequences are immediate: `BOTH_CURRENT` becomes `CPU_CURRENT`, and `GPU_CURRENT` becomes `CONTENTS_LOST`. CPU_CURRENT remains current. Zero-count calls do not change these states.

Device loss or permanent abandonment causes the same buffer transitions. Unknown work completion or incomplete error-protocol results cause quarantine. A confirmed operation-local validation failure with completed error reporting does not by itself establish device loss; safe fallback can proceed if the device remains usable and the old contents can be made current on the CPU.

Apply these changes to buffers in **every instance** that uses the device. The legacy `GPUFailed` latch maps to this same quarantine policy in a runtime that supports both ABIs. A failed legacy run therefore cannot imply that v1 CPU copies are current. The existing implementation has no v1 buffers; this cross-ABI rule is a new requirement.

A full CPU replacement can create a new current version while obsolete GPU allocations remain alive for safe retirement. Charge both until the old allocation is released. A driver hang can delay that release; exhausting the budget then returns `LIMIT_EXCEEDED`, not permission to reuse in-flight storage.

## 10. Status codes

These codes apply only to `wago_gpu_v1`. They do not replace the legacy status table.

| Value | Name | Meaning |
| --- | --- | --- |
| 0 | `OK` | Operation completed |
| 1 | `CPU_FALLBACK` | Dispatch did not commit GPU output; CPU input state is ready |
| 2 | `INVALID_HANDLE` | Unknown, zero, freed, or foreign handle |
| 3 | `INVALID_RANGE` | Invalid index, offset, count, or range overflow |
| 4 | `TYPE_MISMATCH` | Buffer or guest storage has the wrong type |
| 5 | `INVALID_BINDING` | Missing slot, invalid slot, or unsupported alias |
| 6 | `LIMIT_EXCEEDED` | A configured resource limit or allocation limit prevents the operation |
| 7 | `DEVICE_ERROR` | Device operation failed and safe fallback is not available |
| 8 | `CONTENTS_LOST` | The latest committed contents cannot be recovered |
| 9 | `CANCELLED` | The operation was cancelled |
| 10 | `INVALID_KERNEL` | Unknown kernel ID or invalid kernel contract |
| 11 | `UNSUPPORTED_STORAGE` | Unsupported memory or GC storage form |
| 12 | `INVALID_STATE` | Operation is invalid during shutdown or another state transition |

Only `dispatch` returns `CPU_FALLBACK`. An allocation failure MUST become a status when it can be detected before allocation; Go runtime out-of-memory termination is not a recoverable error guarantee. Scalar access traps SHOULD include the corresponding symbolic code and import name in diagnostics.

## 11. Floating-point behavior

A float kernel has any F16/F32/F64 binding, any floating-point local or instruction, or any float access intrinsic. This conservative classification includes conversion-only kernels. Integer-only means none of these is present. The new API keeps explicit consent for relaxed GPU float behavior. Without consent, a float kernel uses CPU fallback. The CPU runner uses Wago's native Wasm arithmetic.

WGSL float evaluation can differ from Wasm in rounding, operation fusion, subnormal handling, and non-finite values. Separate shader statements do not establish strict Wasm arithmetic. See the [WGSL floating-point rules](https://www.w3.org/TR/WGSL/#floating-point-evaluation).

The F16 conversion rules in section 6 are exact for a given input bit pattern. They do not make preceding relaxed GPU arithmetic identical to CPU arithmetic. Tests MUST distinguish conversion correctness from numerical agreement of a whole kernel. Tolerances MUST be stated for each workload and input range.

## 12. Memory limits and resource use

The proposed default limits are:

| Limit | Default |
| --- | --- |
| Live buffers per instance | 64 |
| Selected buffer kernels per module | 64 |
| Slots per kernel | At most 8, further limited by the device |
| Elements per buffer | 10,000,000 |
| Tracked resident allocation per instance | 256 MiB |
| Tracked resident allocation per plugin runtime | 512 MiB |

The host can lower these limits. Device allocation limits can be lower. The compiler's existing bounded source and instruction limits remain in effect unless a later change gives a tested reason to adjust them.

Allocation accounting MUST include CPU copies, expanded GPU storage, upload buffers, readback buffers, temporary outputs, and retained pool capacity. Diagnostics MUST distinguish logical data bytes from physical allocation bytes. Native driver overhead that the binding cannot report MUST be identified separately; these limits are not a promise about total process RSS.

The plugin SHOULD reuse pipelines and buffers within these bounds. It MUST NOT retain an unbounded pool, grow buffers with quadratic copying, or allocate a temporary object for every shader element. Completed CPU scalar access SHOULD avoid per-element heap allocation. A CPU mirror can be allocated lazily if zero initialization and fallback behavior remain correct.

The proposed compiler also has fixed retention limits: 512 KiB of generated WGSL per kernel, 8 MiB per pending compilation, 16 MiB across pending compilations, and 256 cached GPU pipelines per runtime. Stop generation before a byte limit is exceeded. Evict pending WGSL entries first. Pipelines are released when their last live owner leaves; v1 has no idle pipeline cache. If 256 live-owned pipelines prevent a new entry, retain CPU fallback for the new entry. Buffer-allocation budgets do not include unknown native pipeline allocations; the pipeline count bounds those objects separately. These are new limits, not claims about the current scalar cache.

### 12.1 Active compiler admission

The proposed runtime permits **one active heavy compiler task** at a time, covering metadata decoding and GPU lowering for both interfaces. Source hooks wait for this private permit before allocating decoder/value/code-generation workspace. Waiting MUST NOT hold the GPU, instance, or diagnostic lock, retain an extra source copy, or start an extra worker goroutine. Check shutdown admission again after acquiring the permit. Release it on every exit, including a decoder error.

Process selected kernels sequentially under the permit. Per-kernel temporary workspace and generated-source limits remain in force. Charge generated text to active or pending storage exactly once when ownership moves. This bounds active compiler work separately from retained metadata, pending WGSL, and buffer budgets.

If a retention or generation limit prevents GPU lowering after CPU metadata was verified, preserve that contract and select CPU fallback unless an established contract error applies. Compiler contention alone waits; it MUST NOT turn a verified contract into `INVALID_KERNEL`. Standalone `CompileBufferWGSL` keeps its per-call bounds; its caller controls concurrency. Test concurrent source hooks, bounded workspace, failed decoding, and shutdown while a hook waits.

### 12.2 Charging and peak storage

Every live tracked allocation is charged once to the runtime, including legacy scalar storage/readback, all v1 buffers, in-flight retired storage, and safe idle buffer pools. Reservations across instances MUST be atomic. Growth reserves the peak overlap of old and new storage before allocation. Moving a block between an instance and a shared idle pool changes its owner, not its runtime charge. It MUST neither erase nor duplicate that charge.

Instance-owned and retired in-flight allocations also count against that instance's cap. Legacy allocations shared by a cached scalar pipeline count at runtime scope. Shared idle buffer pools count at runtime scope, and acquire an instance charge before active reuse. Reused storage MUST be zeroed or fully initialized before a new guest can observe it. Reclaim safe idle pools before rejecting a reservation, including a CPU-fallback preparation. In-flight allocations are not reclaimable pools.

A naive two-buffer plan at ten million F32 elements can require 280,000,000 bytes: 80 MB of CPU copies, 80 MB of committed GPU data, 40 MB of temporary output, 40 MB of readback, and 40 MB of retained upload storage. That exceeds 256 MiB. The required first plan retains at most one reusable transfer allocation when no concurrent transfer requires more, reclaims safe upload capacity before readback, and releases or reuses retired scratch after completion. These choices can reduce that example to 240,000,000 tracked bytes before small metadata costs. The implementation MUST measure the actual peak, reserve every overlap, and report `LIMIT_EXCEEDED` if its backend requires more. The arithmetic is a capacity plan, not a measured result.

The runtime cap also prevents 256 legacy pipelines from each retaining two 40,000,000-byte buffers without accounting. Their unconstrained total would be 20,480,000,000 bytes. Legacy GPU calls fall back if a proposed runtime reservation fails. Never prepare CPU fallback with stale copies to avoid a budget error.

## 13. Complete proposed WAT example

This module has a real Wasm start function. It creates two buffers, sends `[1, 2, 3, 4]` to the first buffer, applies two different kernels, and copies `[3, 5, 7, 9]` to guest memory at byte offset 16. Each dispatch can fall back independently.

The first kernel computes `2 * x`. The second computes `x + 1`. When both use the GPU, the intermediate buffer stays on the GPU. The export table in section 3 and relaxed float consent are required host configuration.

**This example runs against the buffer plugin.** Its complete host and fixtures are in [examples/buffers/main.go](examples/buffers/main.go). The earlier scalar start example remains in [examples/start/module.wat](examples/start/module.wat).

```wat
(module
  (import "wago_gpu_v1" "createBuffer"
    (func $create (param i32 i32) (result i32 i32)))
  (import "wago_gpu_v1" "freeBuffer"
    (func $free (param i32) (result i32)))
  (import "wago_gpu_v1" "bindBuffer"
    (func $bind (param i32 i32) (result i32)))
  (import "wago_gpu_v1" "getBuffer"
    (func $get (param i32) (result i32)))
  (import "wago_gpu_v1" "setBuffer32"
    (func $set (param i32 i32 i32 i32 i32) (result i32)))
  (import "wago_gpu_v1" "copyBuffer32"
    (func $copy (param i32 i32 i32 i32 i32) (result i32)))
  (import "wago_gpu_v1" "readBufferF32"
    (func $read (param i32 i32) (result f32)))
  (import "wago_gpu_v1" "writeBufferF32"
    (func $write (param i32 i32 f32)))
  (import "wago_gpu_v1" "dispatch"
    (func $dispatch (param i32 i32) (result i32)))

  (memory (export "memory") 1)
  (global $a (mut i32) (i32.const 0))
  (global $b (mut i32) (i32.const 0))

  ;; This small example traps on an error other than CPU fallback.
  ;; Instance cleanup releases any buffer still alive after a trap.
  (func $check (param $status i32)
    (if (local.get $status) (then unreachable)))

  (func $double (export "wago_gpu.kernel.double") (param $i i32)
    (call $write
      (call $get (i32.const 1))
      (local.get $i)
      (f32.mul
        (call $read (call $get (i32.const 0)) (local.get $i))
        (f32.const 2))))

  (func $addOne (export "wago_gpu.kernel.addOne") (param $i i32)
    (call $write
      (call $get (i32.const 1))
      (local.get $i)
      (f32.add
        (call $read (call $get (i32.const 0)) (local.get $i))
        (f32.const 1))))

  ;; This loop executes on Wago's native CPU path.
  (func $cpuLoop (param $count i32) (param $which i32)
    (local $i i32)
    (block $done
      (loop $next
        (br_if $done (i32.ge_u (local.get $i) (local.get $count)))
        (if (i32.eq (local.get $which) (i32.const 1))
          (then (call $double (local.get $i)))
          (else (call $addOne (local.get $i))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $next))))

  (func $cpuDouble (export "wago_gpu.cpu.double") (param $count i32)
    (call $cpuLoop (local.get $count) (i32.const 1)))
  (func $cpuAddOne (export "wago_gpu.cpu.addOne") (param $count i32)
    (call $cpuLoop (local.get $count) (i32.const 2)))

  (func $run (param $id i32) (param $count i32)
    (local $status i32)
    (local.set $status (call $dispatch (local.get $id) (local.get $count)))
    ;; Save dispatch status at byte 32 or 36 for the host example.
    (i32.store
      (i32.add (i32.const 32)
        (i32.mul (i32.sub (local.get $id) (i32.const 1)) (i32.const 4)))
      (local.get $status))
    (if (i32.eq (local.get $status) (i32.const 1))
      (then
        (if (i32.eq (local.get $id) (i32.const 1))
          (then (call $cpuDouble (local.get $count)))
          (else (call $cpuAddOne (local.get $count)))))
      (else (call $check (local.get $status)))))

  (func $start
    (local $status i32)
    (f32.store (i32.const 0) (f32.const 1))
    (f32.store (i32.const 4) (f32.const 2))
    (f32.store (i32.const 8) (f32.const 3))
    (f32.store (i32.const 12) (f32.const 4))

    ;; Type ID 8 is F32. Each buffer contains four elements.
    (call $create (i32.const 8) (i32.const 4))
    (local.set $status)
    (global.set $a)
    (call $check (local.get $status))
    (call $create (i32.const 8) (i32.const 4))
    (local.set $status)
    (global.set $b)
    (call $check (local.get $status))

    ;; Set A from memory 0, byte offset 0, with four elements.
    (call $check (call $set
      (global.get $a) (i32.const 0)
      (i32.const 0) (i32.const 0) (i32.const 4)))

    ;; B = 2 * A.
    (call $check (call $bind (i32.const 0) (global.get $a)))
    (call $check (call $bind (i32.const 1) (global.get $b)))
    (call $run (i32.const 1) (i32.const 4))

    ;; A = B + 1. No guest transfer is needed between kernels.
    (call $check (call $bind (i32.const 0) (global.get $b)))
    (call $check (call $bind (i32.const 1) (global.get $a)))
    (call $run (i32.const 2) (i32.const 4))

    ;; Copy A to memory 0, byte offset 16.
    (call $check (call $copy
      (global.get $a) (i32.const 0)
      (i32.const 0) (i32.const 16) (i32.const 4)))
    (call $check (call $free (global.get $b)))
    (call $check (call $free (global.get $a))))

  (start $start))
```

The plugin MUST install instance buffer state and imports before the start function runs. Wago instantiation runs the declared start function. The Go host does not need to call a later export to produce these results. Merely naming an export `_start` does not create a Wasm start section.

## 14. Wago integration

Use the existing source-transform and compile-observer path to inspect the module and attach compiled kernels to its identity. Leave the Wasm source unchanged. Use the existing instance hooks to create and close per-instance buffer state.

The inspected Wago dependency is commit `7aa401f29a33eb9050559aab5488cdb307f5a078`. Its checked APIs include memory32/memory64 address metadata, indexed memory ranges, numeric GC-array views, and callback-scoped GC references. See [guest_storage.go](https://github.com/wago-org/wago/blob/7aa401f29a33eb9050559aab5488cdb307f5a078/src/wago/guest_storage.go). Its instantiate interceptor runs before the guest start function; see [access.go](https://github.com/wago-org/wago/blob/7aa401f29a33eb9050559aab5488cdb307f5a078/src/wago/access.go).

No GPU-specific Wago compiler or runtime change is needed. However, the pinned **plugin views** do not expose every fact required for final CPU-contract validation. `ModuleView` has no exported-function lookup; `ModuleCompiledEvent` has no final source bytes. `GuestMemoryInfo` has address type and length but no shared flag. Checked range access alone does not reject shared memory.

Wago's public host-side `Compiled` API does expose export signatures, and `PreparedCompile.Source()` exposes final transformed bytes to its owner. These are not the same as metadata available inside the plugin's compile observer. `Runtime.Module` and `Runtime.AdoptModule` do not run source transformers and report a zero source digest.

The preferred generic additions are an export/signature lookup on `ModuleView` and a `Shared` flag in `GuestMemoryInfo` or an indexed memory descriptor. A generic immutable GC-array range-copy API would avoid full-array copies. Effective import-provider inspection would permit independent override enforcement, but v1 instead requires the host rule in section 21. None of these APIs is added by this specification revision.

Until verified final metadata is available, use the restricted pinned-dependency profile in section 22.1. Unverified contracts return an error, not ordinary GPU fallback. GPU code, buffer management, and translation remain in the plugin. The guest invokes CPU fallback, so nested guest invocation from a host callback is unnecessary.

The existing experiment has one lifecycle limit: an abandoned prepared compilation does not give the plugin a cleanup notification. It bounds pending source records to 16. This proposal retains that bound until a tested replacement exists. If Wago later needs an API for this case, a generic prepared-compilation disposal notification would be preferable to a GPU-specific hook.

If implementation finds another public API gap, the change MUST document the exact blocked operation and propose a generic API. It MUST NOT use private runtime memory layouts as a substitute.

## 15. Required tests

CPU-only tests MUST pass without GPU hardware. Hardware tests MUST distinguish a real adapter run from CPU fallback and skipped tests.

| Area | Required cases |
| --- | --- |
| Compiler | Different actual Wasm expressions; operand order; exact import identity and signature; unsupported instructions; invalid kernel exports; bounded decoding |
| Buffer access | Each type; sign and zero extension; narrow write truncation; exact type checks; zero length; final valid index; invalid index |
| Handles | Zero, stale, freed, and foreign handles; exhaustion; unbinding on free; cleanup after start traps |
| Guest memory | wasm32 and wasm64; selected memory index; address-type mismatch; overflow; short source/destination; unchanged destination after failure; shared-memory rejection |
| GC storage | Each numeric storage kind; F16 raw i16 storage; mutable and immutable arrays; null, struct, and reference-array rejection; callback lifetime rules |
| F16 | All 65,536 stored patterns on read; write rounding ties and boundaries; signed zero; subnormals; infinities; canonical NaNs; CPU/GPU conversion agreement |
| GPU data | Two different kernels sharing an intermediate buffer; no extra transfer between successful GPU dispatches; readback once per version; partial-range preservation |
| GPU writes | Adjacent narrow elements written by different invocations; padded workgroups; in-place access through one slot; alias rejection |
| Fallback | Missing device; rejected shader; unsupported selected code; threshold; correct original CPU results; current CPU data before fallback |
| Failure | No partial GPU commit; allocation limits; injected cancellation and device error; stale CPU copy rejected; contents-lost recovery through full replacement |
| Lifetime | Repeated dispatch; instance close; module close with a live instance; runtime close; in-flight cleanup; bounded pipeline and buffer caches |
| Concurrency | Separate instances; serialized GPU submissions; no cross-instance binding or data use; race detector |

Memory64 tests SHOULD combine small real modules with range-check tests for high addresses. They MUST NOT require huge allocations merely to exercise overflow checks.

The new implementation MUST retain the original `2*x` and `x*x+1` CPU/GPU comparisons. It MUST also run the two-kernel example above through CPU-only, all-GPU, and mixed GPU/fallback paths. Test reports MUST say which failures were injected and which occurred on real hardware.

### 15.1 Boundary interaction acceptance tests

These are separate acceptance results. None is implied by a successful arithmetic comparison.

| ID | Test | Required result |
| --- | --- | --- |
| B01 | Evict WGSL, then complete CPU compilation | Validated CPU contract remains; GPU can fall back |
| B02 | Later transform changes export or memory; use Module/AdoptModule | Never use stale metadata; restricted profile rejects unverified contract/storage |
| B03 | Shared and unshared memory with GPU disabled | Same storage validation as GPU-enabled mode |
| B04 | Host override of one intrinsic | Controlled setup rejects it; no claim that name matching proves provider |
| B05 | Delayed native callback under GC/allocation pressure, timeout and close | Valid context until native release; exactly one cleanup |
| B06 | Rejected command, successful completion, then validation error | No output publication |
| B07 | Cancellation before/after commit decision | One terminal result with the defined mutation semantics |
| B08 | Timeout, full CPU replacement, late callback | New CPU version remains; old resources retire safely |
| B09 | Failure in a dispatch with multiple output slots | No subset of output versions commits |
| B10 | Device loss in one instance with GPU-only data in another | Device-wide state transitions and no stale-data fallback |
| B11 | Legacy failure after successful v1 output | Quarantine applies across both interfaces |
| B12 | Pool reuse across instances | New zero-filled buffer exposes no previous data |
| B13 | Large immutable GC source, one-element range, small budget | `LIMIT_EXCEEDED` before whole-payload allocation |
| B14 | Huge expanded locals, deep stack, invalid signed LEB | Bounded rejection before allocation or expansion |
| B15 | Saved local/read, later mutation, multiple local copies, zero locals | Earlier values remain snapshots; index/handle categories are correct |
| B16 | Compiled TinyGo management guest and full chain | Exact signatures and both actual execution paths verified |
| B17 | Runtime close with blocked backend, then release it | No Stop/close-callback deadlock; one release per resource |
| B18 | Transfer failure just before commit; cancellation during final copy | Failure leaves destination unchanged; committed copy reports success |
| B19 | Small prefix of large buffer near resource cap | Correct tail, measured full-copy costs, correct peak accounting |
| B20 | Exhaustive F16 round trip and F16-to-F32 bit copy | Exact conversion-only contract, separate from float tolerance |
| B21 | Prepare source A and try to adopt different artifact B with identical signatures | Controlled host rejects adoption before kernel attachment |
| B22 | Create and close more than 64 unused preparations, then compile fresh source | Fresh CPU metadata can still be verified |
| B23 | Establish contract errors, then evict WGSL, with and without GPU | `INVALID_KERNEL` remains; no downgrade to fallback |
| B24 | Read-only use of READ_WRITE, unused WRITE, and an identical-value actual store | Only actual stores create output versions and seed copies |
| B25 | Cancel outer Wago invocation or trap after successful import | Committed contents remain; outer error does not imply rollback |
| B26 | Constant/index fill, 0xffffffff, local copies, narrow writes, saved-index restoration | Correct data bits; handles cannot be stored as numbers |
| B27 | One exact F32 read feeds arithmetic and a direct-copy output | Materialization does not mutate the exact value record |
| B28 | Rejected submission and delayed error; profiled failure between stages | No confirmed GPU success; unavailable optional times are zero |
| B29 | One-element work in a large buffer plus an implicit scalar readback | Guest, host-device, seed, and aggregate scalar traffic are distinct |
| B30 | Concurrent compiler hooks with decode failure and shutdown | One active workspace; permit released; metadata survives lowering limits |
| B31 | Packed handle 0x80000001; cancelled valid zero-count call | No sign extension; import no-op returns OK without repairing data |
| B32 | Failed Stop or cancelled work context in the complete host | Cleanup starts with its own context; close errors reach the caller |
| B33 | CPU fallback loop includes getBuffer and scalar accesses/traps | Last and its sequence still identify the preceding dispatch |
| B34 | External-package compiler error inspection through wrapping | Public categories distinguish signature, unsupported opcode, and limit |
| B35 | One handle in two declared slots, one unused | INVALID_BINDING for GPU, forced fallback, and zero count |
| B36 | Full-range increment with unrelated temporary data | Old committed input is seeded; coverage alone cannot omit it |

Tests B05–B06 require the repaired or replacement native binding, not only a Go fake. Use deterministic fake-backend tests for ordering and then native callback tests for ownership. Record injected failures separately from physical device events.

## 16. Performance measurements

Measure 1,024; 16,384; 1,000,000; and 10,000,000 elements. Test single kernels and chains. Separate the first run from repeated execution. Also measure a small prefix of a large buffer: a scalar read or partial CPU update can require a full readback, and a small dispatch can seed a whole output. Report device-local seed copies, fresh-input calls, resident chains, cold startup, and Wago prepared-artifact cache eligibility separately.

Record:

- Native Wago CPU time using the same buffer imports and kernel body.
- Native Wago CPU time using direct guest-memory loads and stores for the equivalent workload.
- GPU execution time, upload time, readback time, and total guest-call time.
- Wasm compilation, shader creation, and pipeline creation time.
- Go allocation count and bytes, peak tracked storage, and process memory measurements.
- Logical and physical buffer sizes, transfer bytes, and transfer count.
- GPU model, driver, operating system, Go version, device limits, numerical mode, workgroup size, repetitions, and timing method.

The direct-memory CPU baseline is required because a host call for each CPU element can be costly. A speedup over that overhead alone does not establish a speedup over ordinary native Wasm.

GPU timestamps MUST be used only when the backend supports and validates them. If unavailable, report device-only time as unavailable. A host wait duration MUST NOT be presented as pure GPU compute time. Stage profiling that adds waits MUST be reported separately from normal end-to-end timing.

Do not invent measurements for the proposed API. Existing hardware measurements remain in [IMPROVEMENTS.md](IMPROVEMENTS.md). Future results MUST include slow cases and identify the smallest measured size where GPU execution wins for each workload and device. There is no universal crossover size.

## 17. Implementation order and acceptance

1. **Contract decisions.** Establish final metadata, the import-provider rule, commit/cancellation policy, callback ownership, and the first guest toolchain. Keep the ABI unfrozen until the guest management interface is proven.
2. **CPU F32 contract.** Add handles, bindings, checked memory32 transfers, and native CPU runners. Run the WAT and real guest-language examples.
3. **Fake-backend state tests.** Inject errors, cancelled work, late completion, multiple-output failure, pool reuse, and shutdown before hardware acceptance.
4. **Hardware F32 chain.** Accept only a repaired or replacement backend with safe callbacks and completed error reporting. Verify the two kernels and all CPU/GPU combinations.
5. **Early F32 measurements.** Measure direct-memory CPU, buffer-ABI CPU, cold startup, new inputs, resident chains, seed copies, and small prefixes of large buffers.
6. **Narrow types and exact F16.** Enable each type only after its access and conversion tests pass.
7. **Additional storage.** Complete GC, memory64, and CPU-only 64-bit imports. Move necessary module metadata parsing earlier if the chosen guest compiler emits those types.
8. **Complete v1 validation.** Run the full CPU interface, hardware, compatibility, resource, and benchmark matrix. Mark any earlier build as a development subset.

The proposal becomes an implemented contract only after the required APIs, tests, and examples pass. A hardware claim requires a real GPU run. The report MUST list any subset that still uses CPU fallback, any untested platform, and any remaining public API gap.

The first three priorities are contract validation and dependency gates, the CPU F32 interface, and the fake-backend state tests. Complete v1 requires all 34 CPU imports, the stated supported GPU subsets, and the acceptance evidence. Hardware-specific fallback remains valid; missing CPU interfaces do not constitute complete v1.

## 18. Existing scalar interface

**Existing.** This section specifies the implemented interface. It remains supported when the buffer interface is added.

### 18.1 Guest imports and status

```text
module: wago_gpu

run(inputByteOffset: i32, outputByteOffset: i32, count: i32) -> status: i32

run_batch(inputByteOffset: i32, outputByteOffset: i32,
          count: i32, passes: i32) -> status: i32
```

The host selects one exported Wasm function with signature `(f32) -> f32`. Each input element is passed to that function. `run_batch` applies the same function repeatedly; each pass takes the preceding pass's result. It does not select different functions.

Offsets, counts, and passes are interpreted as unsigned `i32` values. The two byte ranges are in guest memory 0. Each range has length `count * 4`, calculated in `uint64`. Data is little-endian f32 storage. The CPU guest MUST have its own loop for fallback.

| Value | Exported Go constant | Meaning |
| --- | --- | --- |
| 0 | `Success` | Final results copied to the guest, or an eligible zero-count no-op |
| 1 | `Fallback` | No results copied; the guest can execute its original CPU loop |
| 2 | `InvalidRange` | Checked storage or memory-range access failed |
| 3 | `InvalidBatch` | Pass count is outside 1 through 64 |

`run` always uses one pass. `run_batch` validates its pass count before other work. A guest MUST NOT treat every nonzero value as fallback: invalid ranges and invalid pass counts are errors.

### 18.2 Call behavior and limits

The existing execution order is:

1. Capture the arguments before writing a result to shared host-call slots.
2. Validate the pass count. Set the default result to fallback.
3. Resolve the exact caller instance and invocation context.
4. Start the configured timeout and take the plugin lock.
5. Open a checked guest-storage callback. Check memory 0's address type and both complete byte ranges.
6. Reject partial overlap for GPU execution. Apply the element limit and CPU threshold.
7. Check that this instance has a usable compiled pipeline and the GPU has not failed.
8. Complete an eligible zero-count call without submitting work. Otherwise check cancellation.
9. Upload current input, run the passes, read the complete result, and copy it to the checked output range.

An exact in-place range is supported. Partially overlapping input and output use fallback, because a forward CPU loop can have different results from whole-array GPU staging. The CPU implementation retains its own overlap semantics.

The guest-storage borrow spans the GPU call in this implementation. Neither the input slice nor the commit callback survives the call. The buffer API uses owned copies and does not adopt this as a requirement to retain guest views between imports.

An array over `MaxElements`, one below `MinElements`, a missing pipeline, or an unavailable GPU selects fallback after range checks. Therefore an invalid range remains an error when GPU execution is unavailable. Early caller-resolution failure returns fallback without accessing storage. A zero-count call returns success only after the normal threshold and usable-pipeline checks; it can return fallback in a CPU-only build.

A backend execution error leaves guest output unchanged and sets the runtime's `GPUFailed` latch. Later GPU calls use fallback. A cancellation detected before entering the backend does not set that latch. There is no reset operation: make a new plugin and runtime to retry.

### 18.3 Scalar compiler

The public function is:

```go
func CompileWGSL(source []byte, export string) (string, error)
```

It returns generated WGSL or an error. It does not create a GPU device or execute the module. It is a subset decoder, not a complete Wasm validator. An application MUST still validate the complete module through Wago before execution.

The selected export MUST be a defined, unique function with one f32 parameter and one f32 result. Imported kernels and additional local declarations are unsupported. The body accepts only:

| Instruction | Behavior |
| --- | --- |
| `local.get 0` | Push the input f32 |
| `f32.const` | Push a constant using its original 32-bit representation |
| `f32.add` | Pop right and left operands; push left plus right |
| `f32.sub` | Pop right and left operands; push left minus right |
| `f32.mul` | Pop right and left operands; push left times right |
| Final `end` | Require exactly one result and no trailing code |

The compiler emits separate value definitions, preserves operand order, and rejects stack underflow, missing termination, and malformed immediates. Constants use a WGSL bitcast from their original bits. That preserves the constant representation in generated source; it does not override relaxed arithmetic rules.

| Decoder bound | Existing value |
| --- | --- |
| Module source | 4 MiB |
| Entries in a decoded vector | 16,384 |
| Selected function body | 16,384 bytes |
| Decoded kernel instructions | 4,096 |
| Unsigned 32-bit LEB128 immediate | At most 5 bytes, with valid high bits |

The decoder accepts the Wasm version 1 header and MVP section order. Custom sections are skipped. Duplicate, out-of-order, and unsupported standard section IDs are rejected. Type sections must use function types. Imports must be functions. If a memory section is present, it must declare one unshared memory32. Imported memory, memory64, shared memory, multiple memories, GC type sections, and data-count sections are outside this decoder. A module with no memory can be translated, but the scalar run import still needs memory 0.

The generated scalar shader uses one read/write storage binding and a workgroup size of 256. Each invocation processes one f32. It checks the logical bound storage length. A reused buffer's unused capacity MUST NOT extend the logical input range.

## 19. Existing Go host API

**Existing.** The package name is `wagogpu`. The local Go module path is `wago-gpu`.

```go
func New(config Config) (*Plugin, error)
func (p *Plugin) PluginSet() (wago.PluginSet, error)
func (p *Plugin) Register(reg *wago.Registrar) error
func (p *Plugin) Snapshot() Snapshot
```

`Register` is Wago's plugin entry point. Normal applications use `PluginSet` and `Runtime.LoadPlugins`. Each `Plugin` belongs to one runtime. Repeated registration of the same plugin returns an error. The application closes Wago instances, modules, and the runtime; it does not separately close private GPU objects.

| `Config` field | Type | Default and validation |
| --- | --- | --- |
| `KernelExport` | `string` | Required, nonempty; exact export name |
| `RelaxedFloat` | `bool` | False; GPU startup is skipped unless true |
| `Disabled` | `bool` | False; true forces CPU fallback |
| `ProfileStages` | `bool` | False; true adds waits for host stage timing |
| `MaxElements` | `uint32` | Zero means 10,000,000; cannot exceed that value |
| `MinElements` | `uint32` | Zero means always try GPU; cannot exceed `MaxElements` |
| `RunTimeout` | `time.Duration` | Zero means 5 seconds; negative values are invalid |

`New` validates configuration. It does not create a device. Device initialization happens during plugin startup. A device initialization error is recorded and leaves the runtime available for CPU execution. An authority or registration error is a plugin-load error; it is not a GPU fallback result.

The following is the complete host sequence for an already supplied `wasmBytes` slice and the existing scalar interface:

```go
p, err := wagogpu.New(wagogpu.Config{
    KernelExport: "kernel",
    RelaxedFloat: true,
})
if err != nil { return err }
set, err := p.PluginSet()
if err != nil { return err }

rt := wago.NewRuntime()
defer rt.CloseContext(context.Background())
if err := rt.LoadPlugins(context.Background(), set); err != nil { return err }
mod, err := rt.Compile(wasmBytes)
if err != nil { return err }
defer mod.Close()
inst, err := rt.Instantiate(context.Background(), mod)
if err != nil { return err }
defer inst.Close()
// A declared Wasm start function has already run at this point.
```

The import paths are `wagogpu "github.com/jtenner/wago-gpu"` and `wago "github.com/wago-org/wago"`. `context` is from Go's standard library. This fragment belongs in a function that returns `error`. [examples/start/main.go](examples/start/main.go) is a complete runnable host program.

## 20. Proposed Go configuration additions

**Proposed.** Retain the existing constructors and fields. Add the following configuration types and fields; they do not exist in the current package:

```go
type ElementType uint32 // IDs from section 6.
type BufferAccess uint8

const (
    AccessRead BufferAccess = 1
    AccessWrite BufferAccess = 2
    AccessReadWrite BufferAccess = 3
)

type BindingConfig struct {
    Slot   uint32
    Type   ElementType
    Access BufferAccess
}

type KernelConfig struct {
    ID           uint32
    Export       string
    CPUExport    string
    Bindings     []BindingConfig
    RelaxedFloat bool
    MinElements  uint32
}

// Add these fields to Config, alongside the existing fields:
// Kernels                 []KernelConfig
// MaxBuffersPerInstance   uint32
// MaxKernelsPerModule     uint32
// MaxBindingsPerKernel    uint32
// MaxInstanceBufferBytes  uint64
// MaxRuntimeBufferBytes   uint64
```

`KernelExport` selects only the legacy scalar kernel. `Kernels` selects buffer kernels. At least one selection MUST be present. Both forms MAY be enabled together. A buffer-only host is not required to invent a scalar export.

The named Go element constants are `TypeI8`, `TypeU8`, `TypeI16`, `TypeU16`, `TypeI32`, `TypeU32`, `TypeF16`, `TypeF32`, `TypeI64`, `TypeU64`, and `TypeF64`, with IDs from section 6. Version 1 status constants use the `V1` prefix and `int32`, as listed in the canonical ABI table, to avoid collisions with legacy constants.

The top-level `RelaxedFloat` and `MinElements` retain their legacy meaning. Each buffer kernel uses its own values from `KernelConfig`. Float consent is not inherited. `Disabled`, `ProfileStages`, `MaxElements`, and `RunTimeout` apply to both interfaces. An integer-only kernel does not require relaxed float consent.

Zero values for the new resource limits select the defaults in section 12. Version 1 accepts only positive values at or below those defaults after normalization. `MaxInstanceBufferBytes` MUST NOT exceed `MaxRuntimeBufferBytes`. A kernel threshold MUST NOT exceed `MaxElements`. The configured list MUST fit `MaxKernelsPerModule`.

`New` MUST copy configuration slices, including binding slices. Later caller mutation MUST NOT change compilation or execution behavior. Configuration is fixed for the life of the runtime. A different kernel table requires a new plugin.

Kernel IDs MUST be nonzero and unique. Kernel exports and CPU exports MUST use the prefixes in section 3, with a nonempty matching suffix. Duplicate names, duplicate slots, unknown element types, and access values other than 1, 2, or 3 are invalid configuration. Slots are numbered from zero and MUST be below `MaxBindingsPerKernel`; gaps are allowed. Version 1 requires at least one declared slot per buffer kernel. Each slot used by compiled code MUST be declared. Extra declared slots are still validated at dispatch.

All selected entries apply to each module compiled in this runtime. For example, two independently produced modules can share a runtime if both use ID 1 for `wago_gpu.kernel.double` with its declared slots and runner. A second module that uses ID 1 for a different contract must use a different agreed table or a separate runtime. This runtime-wide configuration does not infer per-module ID meanings.

A missing export or wrong signature marks that entry invalid for that module. `dispatch` returns `INVALID_KERNEL` for it. The error MUST NOT prevent Wago from compiling valid CPU code in that module. An existing valid export with unsupported GPU instructions is different: that entry can return `CPU_FALLBACK`.

A hardware device with lower limits can reject a particular GPU kernel and retain CPU fallback. It does not reduce the logical CPU slot table. No adapter selector, workgroup tuner, environment-variable override, or automatic threshold tuner is part of this proposed Go API.

### 20.1 Proposed compiler entry point

Add a pure translation function for the new kernel form:

```go
func CompileBufferWGSL(source []byte, kernel KernelConfig) (string, error)
```

It validates the supplied kernel contract, locates the named exports, and returns generated WGSL for the supported subset. It does not open a device, allocate buffers, execute Wasm, or validate every unselected function. Wago remains the full module validator. Direct Go errors retain a typed category for invalid contract, unsupported translation, or decoder/resource limit. The plugin uses the classification in section 7.3; it MUST NOT turn every translation error into fallback.

Failures expose this public Go API:

```go
type CompileErrorKind uint8
const (
    CompileInvalidContract CompileErrorKind = iota + 1
    CompileUnsupported
    CompileLimit
)
type CompileError struct {
    Kind CompileErrorKind
    Message string
}
func (e *CompileError) Error() string
```

Use `errors.As(err, &compileErr)` where `compileErr` is `*wagogpu.CompileError`. Wrapping with `%w` preserves the category. Text is diagnostic only. All CompileBufferWGSL failures use this type. Malformed or unverifiable kernel metadata is invalid contract; an unsupported body is unsupported; decoder and configured resource bounds use CompileLimit. Test all categories and wrapping from an external wagogpu_test package.

`RelaxedFloat` and `MinElements` control runtime selection. They do not prevent a caller from inspecting generated shader text with this function. The caller MUST NOT interpret successful translation as strict numerical equivalence, device support, or a completed GPU run.

### 20.2 Complete proposed buffer host

This is a complete **proposed** Go host for the WAT module in section 13. It now builds against the plugin. The runnable copy is [examples/buffers/main.go](examples/buffers/main.go). Compile that WAT as `module.wasm` beside the future host. The WAT saves each dispatch status at byte 32 or 36 before it takes the fallback branch.

```go
package main

import (
    "context"
    "errors"
    "flag"
    "fmt"
    "math"
    "os"

    wago "github.com/wago-org/wago"
    wagogpu "github.com/jtenner/wago-gpu"
)

func main() {
    firstCPU := flag.Bool("first-cpu", false, "force the first kernel to CPU")
    secondCPU := flag.Bool("second-cpu", false, "force the second kernel to CPU")
    requireGPU := flag.Bool("require-gpu", false, "require both hardware dispatches")
    flag.Parse()
    if err := run(*firstCPU, *secondCPU, *requireGPU); err != nil {
        fmt.Fprintln(os.Stderr, err)
        os.Exit(1)
    }
}

func run(firstCPU, secondCPU, requireGPU bool) (resultErr error) {
    if requireGPU && (firstCPU || secondCPU) {
        return fmt.Errorf("require-gpu conflicts with a forced CPU kernel")
    }
    threshold := func(cpu bool) uint32 {
        if cpu { return 5 } // This example dispatches four elements.
        return 0
    }
    slots := []wagogpu.BindingConfig{
        {Slot: 0, Type: wagogpu.TypeF32, Access: wagogpu.AccessRead},
        {Slot: 1, Type: wagogpu.TypeF32, Access: wagogpu.AccessWrite},
    }
    p, err := wagogpu.New(wagogpu.Config{
        Kernels: []wagogpu.KernelConfig{
            {ID: 1, Export: "wago_gpu.kernel.double",
                CPUExport: "wago_gpu.cpu.double", Bindings: slots,
                RelaxedFloat: true, MinElements: threshold(firstCPU)},
            {ID: 2, Export: "wago_gpu.kernel.addOne",
                CPUExport: "wago_gpu.cpu.addOne", Bindings: slots,
                RelaxedFloat: true, MinElements: threshold(secondCPU)},
        },
    })
    if err != nil { return err }
    set, err := p.PluginSet()
    if err != nil { return err }
    wasm, err := os.ReadFile("module.wasm")
    if err != nil { return err }
    ctx := context.Background()
    rt := wago.NewRuntime()
    // Cleanup must not inherit cancellation from the work context.
    cleanupCtx := context.Background()
    defer func() { resultErr = errors.Join(resultErr, rt.CloseContext(cleanupCtx)) }()
    if err := rt.LoadPlugins(ctx, set); err != nil { return err }
    module, err := rt.Compile(wasm)
    if err != nil { return err }
    defer func() { resultErr = errors.Join(resultErr, module.Close()) }()
    // No import overrides are supplied.
    instance, err := rt.Instantiate(ctx, module)
    if err != nil { return err }
    defer func() { resultErr = errors.Join(resultErr, instance.Close()) }()
    expected := [4]float32{3, 5, 7, 9}
    for i, want := range expected {
        got, ok := instance.ReadFloat32Le(16 + uint32(i)*4)
        if !ok || math.Float32bits(got) != math.Float32bits(want) {
            return fmt.Errorf("incorrect final element %d: %v", i, got)
        }
    }
    forced := [2]bool{firstCPU, secondCPU}
    for i := 0; i < 2; i++ {
        status, ok := instance.ReadUint32Le(32 + uint32(i)*4)
        if !ok || status > 1 {
            return fmt.Errorf("invalid saved status for kernel %d", i+1)
        }
        if forced[i] && status != 1 {
            return fmt.Errorf("kernel %d did not use the requested CPU path", i+1)
        }
        if requireGPU && status != 0 {
            return fmt.Errorf("kernel %d used fallback", i+1)
        }
        fmt.Printf("kernel %d: status %d (0=GPU, 1=CPU fallback)\n", i+1, status)
    }
    fmt.Println("verified:", expected)
    return nil
}
```

Test CPU/CPU, GPU/GPU, GPU/CPU, and CPU/GPU separately. The mixed hardware tests MUST assert the unforced dispatch actually used the GPU; the ordinary demonstration permits fallback when hardware is absent. Repeat the same data-flow tests at counts 0, 1, 255, 256, 257, 511, and 512 with a parameterized fixture and adjusted thresholds. The four-element module itself is fixed-size. A zero-count result must not be counted as a hardware run.

Cleanup runs in instance, module, runtime order. The named result joins every returned cleanup error with any work error. An injected Stop error must make this example fail. The independent background cleanup context is intentional: a cancelled work context must not prevent shutdown from starting. It does not provide a hard driver-hang timeout.

### 20.3 First guest-language target and ABI gate

The first source-language target is **TinyGo 0.42.0**, using Go source, `wasm-unknown`, no scheduler, trap panics, the leaking GC mode, optimization level 2, and no debug information. The checked probe is [spec/guest-tinygo.go.txt](spec/guest-tinygo.go.txt). It is stored as text so host `go test ./...` does not build Wasm-only declarations.

```sh
cp spec/guest-tinygo.go.txt /tmp/wago-gpu-guest-probe.go
tinygo build -target=wasm-unknown -scheduler=none -panic=trap   -gc=leaking -opt=2 -no-debug -o /tmp/wago-gpu-guest-probe.wasm   /tmp/wago-gpu-guest-probe.go
wasm-validate /tmp/wago-gpu-guest-probe.wasm
wasm2wat /tmp/wago-gpu-guest-probe.wasm -o /tmp/wago-gpu-guest-probe.wat
```

During this revision, that source compiled and its imports/exports were inspected. The selected kernel uses `i32.const`, `local.get`, `local.tee`, `f32.add`, direct buffer intrinsic calls, and final `end`. TinyGo changed source multiplication by two into addition in the actual Wasm. The GPU compiler must compile those actual instructions. No shader is chosen from the source expression.

Use TinyGo's `//export` for this tested kernel form. A separate probe with `//go:wasmexport` emitted an initialization guard with a memory load and control flow, outside the GPU subset. It must be rejected, not stripped without proof. The guest host must follow the toolchain's initialization requirements, including `_initialize` where applicable; that export is not a Wasm start section.

The probe proves syntax, signatures, and a compatible selected body. It does **not** prove complete guest runtime support, CPU buffer execution, GPU execution, or all management imports. Before freezing v1, expand it into a full allocation/transfer/dispatch/free guest and run both paths. Any required module-inspection support moves before that milestone.

Standard Go 1.27.1 restricts Wasm import declarations to at most one result; the packed companion avoids that management signature limit. It does not prove that standard Go's exported kernel wrappers fit the subset. Standard Go guest GPU support remains unclaimed until a separate compiled example passes. See [Go's Wasm ABI checks](https://github.com/golang/go/blob/go1.27.1/src/cmd/compile/internal/ssagen/abi.go).

## 21. Plugin registration and authority contract

**Existing and proposed as marked.** The definition is experimental plugin `github.com/jtenner/wago-gpu`, version `0.0.0`, name `Wago GPU`, with MIT provenance at `https://github.com/jtenner/wago-gpu`. The owned repository identity replaces the local placeholder. This packaging change does not alter guest import names.

The plugin requires these Wago authorities:

| Authority | Purpose |
| --- | --- |
| `AuthorityHostImportDefine` | Register the explicit guest interface |
| `AuthorityHostCallerIdentify` | Resolve caller identity and invocation context |
| `AuthorityModuleSourceTransform` | Inspect Wasm bytes without changing them |
| `AuthorityModuleCompileObserve` | Associate shaders with successful compilation and handle compile errors |
| `AuthorityModuleCloseObserve` | Release module ownership |
| `AuthorityInstanceInstantiateIntercept` | Install state before guest start |
| `AuthorityInstanceCloseObserve` | Release instance ownership |

The existing import authority scope is exactly `wago_gpu`. The proposed definition MUST also include `wago_gpu_v1`. It MUST NOT request authority for unrelated import modules. `PluginSet` calculates Wago's definition digest and supplies explicit grants for the declared authorities. The host opts into those grants when it loads that set.

Import module names, field names, parameter types, result types, and result order are exact. Case changes are different names. A wrong signature is an import-link error through Wago, not a dispatch fallback. Modules can import only the functions they use.

Registration MUST resolve callers through Wago. A guest-supplied handle or kernel ID MUST NOT select another instance's state. The plugin MUST NOT infer identity from guest memory, a module filename, or an export name. CPU storage imports MUST remain available when device creation fails or GPU execution is disabled.

The lifecycle stop operation MUST be safe to call more than once. Registration and authority failure MUST not leave a running polling task or a device allocation with no owner. The implementation MUST follow Wago's lifecycle order and release any partially created private resources.

For an instance that can execute GPU kernels, the host MUST NOT override any import the compiler treats as an intrinsic. The pinned runtime permits explicit instance overrides to take precedence for these module names; they are not reserved built-ins. Matching a name and signature does not prove the effective provider. The section 20.2 host passes no overrides. A future setup helper MUST reject conflicting override options. Independent plugin enforcement would require a public effective-provider query; it is not assumed here.

`Stop` MUST NOT depend on terminal instance-close callbacks to finish. Pinned Wago can deliver them after plugin Stop. Close admission and initiate cleanup from private ownership records. Keep state valid for late close callbacks. Stop MAY wait for independent private work, but MUST NOT hold a lock that work needs to finish. Each resource is released exactly once. See the pinned [instance-close ordering](https://github.com/wago-org/wago/blob/7aa401f29a33eb9050559aab5488cdb307f5a078/src/wago/access.go#L317).

## 22. Compilation and resource lifetime

### 22.1 Required compilation sequence

**Existing sequence; proposed multiple-kernel extension.**

1. Receive `CompilationIdentity` and the exact source bytes through the source transformer.
2. Inspect only explicitly selected exports with the bounded decoder. Generate WGSL or a per-kernel rejection reason. Return the original Wasm bytes unchanged.
3. Record the source digest and temporary translation result under that compilation identity.
4. Let Wago validate and compile the complete module.
5. On successful compilation, compare the final source digest with the inspected digest. Do not attach a shader if they differ or if the temporary result is missing.
6. Attach independently validated contract records to the final module identity, then resolve or create device pipelines. Pipeline failure does not remove the CPU contract or undo successful CPU compilation.
7. On compilation failure, discard temporary state and any module state already attached by an earlier observer.

A pipeline MUST NOT become callable merely because source inspection succeeded. A shader from one compilation MUST NOT attach to a different compilation, even if exported names match.

Temporary translations are bounded to 16 pending compilations in the current implementation. Eviction chooses safe fallback if the compilation later completes. The proposed implementation MUST also bound pending bytes, using the source, code, and kernel-count limits. It MUST NOT retain full source copies when a digest and bounded generated result are sufficient.

The existing pipeline cache uses the generated shader string as a key within one device. The proposed key adds the contract information in section 7. Cache entries MUST NOT cross devices. No persistent on-disk cache, binary shader import, or shared global device cache is specified.

### 22.1.1 Final metadata and the pinned-dependency profile

Maintain separate records:

| Record | Contents | Lifetime |
| --- | --- | --- |
| Validated module contract | Final module identity/digest, export signatures, memory declarations including shared flags, and all established contract violations | Module and all live instances |
| GPU translation | WGSL, unsupported-capability/lowering details, actual-use sets, and generation metadata | Bounded; independently evictable; never the sole owner of a contract violation |
| Device pipeline | Validated device resources and cache key | Live owners on one device |

Evicting WGSL MUST NOT evict a validated CPU contract. Run a bounded metadata pass independently of instruction lowering. A valid module outside the GPU instruction subset can still have a validated contract. A module outside the metadata parser's supported forms or bounds cannot.

With the current Wago pin, verified v1 contracts require `Runtime.Compile` or `PreparedCompile.Compile`, a matching nonzero final source digest, and a successful metadata pass. The host MUST NOT use `PreparedCompile.Adopt` for an instance that relies on verified v1 kernel or memory metadata unless a separately specified and tested mechanism proves that the artifact corresponds to the inspected source. No such mechanism is defined in this draft. Reject final-contract use after a source mismatch, missing metadata, zero digest, `Runtime.Module`, or `Runtime.AdoptModule`. Dispatch then returns `INVALID_KERNEL`; memory transfers return `UNSUPPORTED_STORAGE` if their memory declaration cannot be verified. Device-disabled mode uses the same rules. Handle creation and operations that need no module-storage facts can still work. Never silently substitute earlier metadata or a shared-memory assumption.

Keep pending CPU metadata separately from the 16-entry WGSL eviction policy. Bound it to 64 preparations and 64 KiB per preparation. When admitting a new record at the count or byte limit, evict the oldest unattached pending CPU record first. Bound the total to 4 MiB. A preparation that completes after eviction is unverified. Metadata attached to a module or retained by a live instance MUST NOT be evicted.

Represent unverified completion by an absent pending record; do not keep an unbounded set of rejected identities. A record that exceeds its own size limit is not retained. Normal creation and closure of abandoned preparations MUST NOT permanently prevent fresh compilations from being verified, even though the pin emits no abandonment notification. CPU contract records attached to live owners remain until those owners close. Generic metadata APIs could remove this restricted profile later without changing the guest ABI.

`PreparedCompile.Adopt` checks artifact admission limits, then uses the preparation's source digest in `finishCompile`; it does not prove source/artifact equivalence. A matching digest alone cannot distinguish this route. The compile-route restriction is a **host obligation**, like the no-intrinsic-override rule. Controlled host setup MUST expose only the allowed compile routes and reject artifact adoption before attachment. Test source A (`2*x`) and artifact B (`3*x`) with identical signatures. See [PreparedCompile.Adopt and finishCompile](https://github.com/wago-org/wago/blob/7aa401f29a33eb9050559aab5488cdb307f5a078/src/wago/runtime.go#L662).

The pinned plugin-view limits are visible in [hooks.go](https://github.com/wago-org/wago/blob/7aa401f29a33eb9050559aab5488cdb307f5a078/src/wago/hooks.go#L71), [module_view_types.go](https://github.com/wago-org/wago/blob/7aa401f29a33eb9050559aab5488cdb307f5a078/src/wago/module_view_types.go), and [guest_storage.go](https://github.com/wago-org/wago/blob/7aa401f29a33eb9050559aab5488cdb307f5a078/src/wago/guest_storage.go#L32). A host metadata adapter is another possible solution, but no untested adapter is assumed in this release contract.

### 22.1.2 Prepared compilation and artifact caches

At the pin, registering any source-transform hook makes `PreparedCompile.Cacheable()` false, even if it returns unchanged bytes. This affects Wago artifact caching, not the plugin's own pipeline cache. Record cache eligibility in compilation measurements. See [runtime.go](https://github.com/wago-org/wago/blob/7aa401f29a33eb9050559aab5488cdb307f5a078/src/wago/runtime.go#L598).

The host MUST consume or close every prepared compilation. Evicting a plugin record cannot close a host-owned preparation or release its Wago admission resources. The missing abandonment event remains a separate cleanup improvement.

### 22.2 Ownership rules

| Resource | Owner and release point |
| --- | --- |
| Temporary translation | Pending compilation; release on success, error, eviction, or stop |
| Module kernel record | Compiled module; retain required records for live instances after module close |
| GPU pipeline | Reference-counted cache; release after the last module/live-instance owner leaves |
| Buffer handle and bindings | One instance; invalidate on free or instance close |
| CPU/GPU buffer storage | Buffer owner, plus any in-flight submission; release only when both are finished |
| Temporary dispatch output | Current dispatch or bounded reuse pool; never expose it before commit |
| Device, queue, and adapter | One plugin runtime; release after dependent resources |

Closing a module prevents new use through that module according to Wago's rules, but MUST NOT destroy pipelines used by an existing instance. Closing an instance removes its bindings and frees all owned buffers. Closing the runtime releases all remaining state, including modules that the host did not close separately.

A start-function trap is a failed instantiation. Proposed buffer resources created before that trap MUST be released through the failed-instantiation cleanup path. This path requires a specific test; successful-instance cleanup alone is not sufficient evidence.

The implementation MUST NOT release a device before its pipelines and buffers. It MUST NOT free memory retained by a pending native callback. Late completion callbacks MUST be able to finish without blocking on an abandoned receiver. Callbacks MUST NOT capture guest views or call into a closed guest.

Do not make this release order depend on terminal instance-close observers: Wago can deliver those after Stop. Stop begins retirement through its own records and retains the state needed by late observers. Waiting for private backend work must release any lock required by its completion path. If safe retirement cannot complete because the native driver never responds, report that shutdown limit; do not claim a bounded successful cleanup or free callback storage early.

The host SHOULD use `Runtime.CloseContext` when it needs to wait for cleanup. The existing `Runtime.Close` starts asynchronous shutdown. Neither form establishes a hard limit on a native driver hang.

## 23. GPU backend and generated shader requirements

### 23.1 Device support

**Implemented.** The backend is built with `webgpu && cgo && linux`. It uses the local repaired `github.com/oliverbestmann/webgpu/wgpu` v1.36.0 binding and pinned native patch described in section 31.2. Other builds use the CPU fallback stub. The binding uses C glue and CGO; the native repair is Rust. Wago remains unchanged.

The backend requests the default native adapter with a high-performance preference. It accepts integrated or discrete hardware adapters and rejects software adapters. It requests a device, reads its limits, and records adapter and driver information. Only Linux/amd64 with the recorded NVIDIA Vulkan device has hardware evidence in this project.

**Proposed requirement.** Device selection MUST never label software execution as a real GPU result. Missing features and limits MUST be checked before submission. Device initialization MUST consider eligible buffer kernels as well as the legacy float setting; a false legacy `RelaxedFloat` value must not disable an eligible integer buffer kernel.

The original cogentcore wrapper wired `DeviceLostCallback`, but the original backend did not register it. Source review found unsafe callback contexts. The repaired backend now registers loss notification with pinned context storage. Device timestamps remain unavailable. A replacement backend MUST establish host/device memory visibility, callback lifetime, feature negotiation, limits, and completed error reporting. Arithmetic examples alone are insufficient evidence.

### 23.2 Scalar device storage

**Existing.** Each cached scalar pipeline retains one storage buffer and one mapped-readback buffer. Capacity grows geometrically, up to 40,000,000 bytes per buffer and the applicable device limits. A smaller call can retain larger capacity. Each call binds only its logical byte length, uploads current input, and downloads final output.

A batch uses separate compute passes in submission order. It has one upload and one final readback regardless of pass count. It does not retain logical guest data between separate imports. Pipeline and physical allocation reuse MUST NOT be mistaken for input-data reuse.

### 23.3 Buffer-kernel layout

**Proposed.** Use bind group 0. Binding 0 is a `var<uniform>` block of four u32 words, 16 bytes in total: logical count at byte 0 and zero-filled reserved words at bytes 4, 8, and 12. The data is encoded little-endian. The block has no dynamic offset. Slots in the union of actual read and write sets, sorted by logical slot number, map densely to bindings 1 through N. Unused declarations have no physical binding but remain subject to dispatch validation. Record the mapping and both actual-use sets in pipeline metadata.

Check uniform binding size/alignment and uniform-buffer count separately from storage-buffer count, storage binding size, total bindings per group, and compute dispatch/workgroup limits. Do not count the uniform block as another storage buffer. READ maps to `var<storage, read>`. WRITE and READ_WRITE map to `var<storage, read_write>`; the compiler rejects guest reads from logical WRITE slots even though WGSL permits them physically.

For slots actually used by the shader, membership in the actual write set selects storage. Actual outputs refer to temporary storage until commit; slots only read refer to committed storage, even if declared READ_WRITE. Unused declared slots need no output allocation or transfer. To preserve unwritten elements, the first implementation MUST seed each temporary output with the old committed contents. It MAY skip that copy only if it proves that every required element is written, preserves all elements outside the dispatch range, and proves that no output read needs pre-dispatch contents. Full coverage alone does not permit omitting a READ_WRITE seed. Store-first, read-later use can qualify. WRITE does not mean discard the whole buffer.

A READ_WRITE slot in the actual write set uses the same temporary storage for all its accesses within a dispatch. A later read of the same element observes an earlier write by that invocation. The compiler MUST preserve this order. It MUST NOT redirect those reads to an immutable pre-dispatch copy.

One invocation handles one index. Workgroup size is 256 in the first version, with a one-dimensional dispatch of `ceil(count / 256)` groups. A device that cannot support this shape selects fallback. Count arithmetic MUST use a width that cannot overflow. Version 1 does not split one dispatch across multiple devices or perform automatic chunking.

Buffer stride is four bytes for all GPU-supported element types in version 1. The layout is internal; it is not a guest-address layout. Signed reads use the declared width for sign extension. Narrow writes clear the unused high bits. F32 accesses preserve the chosen bit representation at transfer boundaries. No shader receives a guest-memory address or GC reference.

Different dispatches use ordered submissions and valid storage dependencies. Intra-kernel barriers, workgroup shared storage, atomics, subgroup operations, and cross-index communication are outside the supported subset.

### 23.4 Completion and commit

**Proposed.** A GPU buffer dispatch can finish without copying all outputs to the CPU. Commit requires four completed checks: successful resource creation; valid command encoding and submission; successful work completion; and completed relevant validation/allocation/internal-error reporting with no invalidating error. Unknown status fails. Queue completion alone proves none of the earlier validation decisions.

Use per-operation error scopes for every error class the native backend exposes, covering allocation, encoding, submission, and transfers, plus device-loss/error monitoring. Await both work completion and all error-scope results, including explicit no-error completion. Serialize scope ownership with submissions. A backend with synchronous equivalents must document and test their scope and lifetime guarantees. An uncaptured-error listener alone is not a completed error check. The pinned wrapper does not expose enough of this protocol; repair or replace it before accepting this phase. The distinction follows the [WebGPU error handling design](https://gpuweb.github.io/gpuweb/explainer/#error-scopes).

A decisive test rejects a command, reports successful queue completion, then delivers its validation error. The seeded output MUST NOT be published. Test a late error for each output and transfer stage. Completion and every error result are inputs to the single commit decision in section 9.4.

On a nonzero success, exactly the buffers in the actual write set switch to new committed versions as one logical dispatch commit under the instance lock. No other buffer receives a new version or loses a current copy. On failure, none switches. Temporary storage cannot return to a pool until in-flight use is complete. Driver-level device loss can still destroy old GPU-only versions, as section 9 states.

Readback MUST include the required device-to-host synchronization. Map completion and range length MUST be checked. Conversion from physical GPU storage to canonical CPU storage finishes before that CPU version is published. No successful read may observe a partially converted buffer.

### 23.5 Confirmed callback defect and backend acceptance gate

Historical source inspection confirmed that the original cogentcore binding's `Queue.OnSubmittedWorkDone` and `Buffer.MapAsync` pass the address of a local `cgo.Handle` to native asynchronous callbacks without pinning or C-owned context storage. The C bridge forwards that address. The handle roots its associated value, not the separate Go variable whose address is retained. This violates the retained-pointer rule; no crash was reproduced in this review. See [queue.go](https://github.com/cogentcore/webgpu/blob/v0.23.0/wgpu/queue.go#L43), [buffer.go](https://github.com/cogentcore/webgpu/blob/v0.23.0/wgpu/buffer.go#L60), and [Go's handle lifetime rule](https://go.dev/pkg/runtime/cgo/#Handle).

Before GPU acceptance, use a repaired binding with C-owned callback context or explicitly pinned Go context. Pair handle deletion, context release, and unpinning with proof that native access has ended. Timeout, instance close, or guest cancellation alone is not that proof. If a callback will not run, backend cancellation/destruction must provide an equivalent ownership acknowledgment. Otherwise retain and charge the context until safe retirement. A buffered Go channel does not fix an invalid native callback pointer.

Review all completion, error-scope, adapter, and device-loss callback paths, including wrappers that delete a handle on Go return while assuming an error callback is synchronous. Test delayed callbacks under GC and allocation pressure, timeout, instance close, and runtime close. Use Go pointer-check modes where available. Test exactly one release and no access after release.

The original `Queue.Submit` returned a submission index without a general error result. Its scoped wrappers did not provide a complete asynchronous no-error completion interface. The backend must expose the completed error protocol in section 23.4; polling an idle queue is not a substitute.

The [current upstream README](https://raw.githubusercontent.com/cogentcore/webgpu/main/README.md) states that CogentCore's repository will not receive updates and points to `oliverbestmann/webgpu`. That review led to the local oliverbestmann fork and native patch in section 31.2. Current test evidence and remaining limits are in BUFFER_REPORT.md. Retain historical measurements, but do not use them to waive callback or error-reporting tests.

## 24. Detailed operation semantics

**Proposed.** These rules resolve cases not fixed by the signatures alone.

### 24.1 Error order and validation

For a status-returning import, use this order where applicable:

1. Resolve the caller and check that the plugin and instance are active. A resolvable closing instance returns `INVALID_STATE`; an unresolvable caller is a host-call trap.
2. Validate IDs, handles, type codes, slot numbers, and the selected kernel contract.
3. Validate storage kind and mutability, declared bindings, alias rules, and both complete ranges.
4. Check configured limits. For a valid zero-count dispatch or bulk transfer, return `OK` here, before contents acquisition or a cancellation check.
5. For other operations, check cancellation before allocation or GPU work, acquire current data, perform the operation, and commit.

If a call has more than one error at the same step, any applicable code in that step is allowed. No destination may change on such an error. GPU absence is considered after contract validation; it MUST NOT hide an invalid handle or range.

An unknown element type is `TYPE_MISMATCH`. A wrong memory address type, shared memory, or unsupported GC storage form is `UNSUPPORTED_STORAGE`. A numeric GC array of the wrong element storage kind is `TYPE_MISMATCH`. An immutable GC destination is `UNSUPPORTED_STORAGE`. A missing memory index or an out-of-bounds element/byte range is `INVALID_RANGE`. These rules also apply to zero-count transfers.

After contract and limit checks, a zero-count transfer returns `OK` before checking cancellation, without obtaining buffer contents or changing versions. A zero-count dispatch follows section 5 and returns a no-op result even when no device exists. `freeBuffer` does not need readable contents, so it can free a `CONTENTS_LOST` buffer.

### 24.2 CPU access and copies

Scalar CPU writes use the declared storage width. They increment the content version and invalidate any older GPU copy. They require current contents, even for a one-element lost buffer. Only the full-range setBuffer operations specified in section 9.2 guarantee repair of lost contents.

A bulk source is copied during its host call. The guest MUST NOT mutate source or destination storage concurrently through another agent, memory import, or native access while the call is active. Shared Wasm memory is rejected. A host MUST obey Wago's own memory and instance concurrency rules.

**Prepare stage:** check and reserve required storage before changing a destination. A CPU-current destination can be updated in place after all fallible validation and preparation succeeds. If readback, conversion, or allocation can still fail, use owned temporary storage and commit only when it is complete. This rule does not require an extra whole-buffer copy for every successful CPU write.

**Commit stage:** after all fallible preparation and a final cancellation check, update the validated destination without a recoverable failure halfway through. Cancellation during the final copy does not cause a failure result after mutation; finish and return success. For copy-back, obtain and convert the result before changing guest bytes. The final guest-storage callback MUST recheck the destination and then perform the copy. A memory size change between separate callbacks MUST NOT invalidate an earlier unchecked pointer. GC references MUST be resolved in the callback where they are used; no cached native array address is allowed.

GC array element bits define the transfer value. The plugin MUST account for Wago's documented element representation and host byte order; it MUST NOT assume that an arbitrary numeric GC array is a packed f32 slice. F16 uses i16 bits, as specified earlier. The API does not convert an f32 GC array to an F16 buffer implicitly.

### 24.3 Bindings and access modes

`bindBuffer` validates the slot and handle, but does not transfer data. Rebinding does not free the previous buffer. Clearing an already empty valid slot succeeds. `freeBuffer` on a freed handle returns `INVALID_HANDLE`; it is not a second successful free.

Bindings belong to the instance and persist until changed, cleared, freed, or closed. A dispatch takes one stable binding snapshot. It MUST NOT mix bindings from two host calls. Slot access declarations constrain compiled kernel use; they do not make a buffer permanently read-only for unrelated CPU calls or other kernels.

`getBuffer` on the CPU returns the opaque handle. In a shader it selects the statically resolved resource. It does not return an integer GPU address. A shader cannot compare, export, store, or manufacture handle values.

Version 1 exposes no buffer-size query import. The guest knows the type and count it supplied to `createBuffer`. The host and plugin retain those values for validation. Imported functions for resize, reinterpretation, sub-buffer views, and handle sharing are not implied.

### 24.4 Fallback decision and cancellation

Check a kernel's CPU threshold once for the whole dispatch. A threshold is a host policy, not a measured promise. The plugin does not change it from past call times. Unsupported GPU instructions or CPU-only element types select fallback after valid bindings are established.

If GPU scratch allocation or a device limit prevents execution, the plugin MAY return `CPU_FALLBACK` only if it can make the entry-state CPU copies current within the configured limits. If that preparation itself exceeds a limit, return `LIMIT_EXCEEDED`. If the only current contents are lost, `CONTENTS_LOST` takes precedence over fallback.

The invocation context and `RunTimeout` bound polling for dispatch and every GPU-dependent transfer, including implicit readback from a scalar read or partial scalar write. The timeout starts at import entry and includes queue-lock waiting, but version 1 need not interrupt a blocking mutex acquisition. Check the context again after acquiring the lock and before submission. Cancellation does not grant permission to use stale data or publish an unconfirmed output.

Use the single commit decision in section 9.4. It occurs immediately before publication or the final non-failing copy, not after mutation. Cache changes and storage retirement can continue after a terminal result, but callbacks cannot change contents or that result.

## 25. Diagnostics and measurements

### 25.1 Existing snapshot

**Existing.** `Snapshot()` takes the plugin lock and returns a value copy. It can wait for a running GPU call. It is a diagnostic summary, not a complete per-instance event log.

| Field | Meaning |
| --- | --- |
| `Device` | Adapter, driver, type, and backend description when available |
| `Reason` | Last recorded rejection or failure text; it can remain after a later success |
| `GPUFailed` | Backend execution has disabled further GPU use in this runtime |
| `DeviceInit` | Host wall time for device initialization |
| `Build.Translate`, `Build.Pipeline` | Most recent observed build times; a cache hit has no new pipeline build |
| `Last.Upload`, `Compute`, `Download` | Host stage durations, populated by stage profiling |
| `Last.Commit` | Final guest copy duration when a backend run completes |
| `Last.Total` | Plugin call time after acquiring its lock, including range checks and commit |
| `Last.DeviceCompute` | Zero while device timestamps are unavailable |
| `Last.TimestampsValid` | False in the current backend |
| `Last.UploadBytes`, `DownloadBytes` | Application buffer bytes transferred by that backend call |
| `Last.Passes` | Requested pass count for an entered backend run |
| `Successes`, `Fallbacks` | Calls counted after caller resolution and entry into the main call path |
| `Modules`, `Instances`, `Pipelines`, `Pending` | Current tracked resource counts |
| `BufferBytes` | Retained storage and readback capacity across cached scalar pipelines |

Early invalid-batch or caller-resolution exits need not update `Last` or the counters. Invalid-range results are not fallback counts. A zero-count success can increment `Successes` without GPU work. Thus the existing aggregate counter alone is not proof of a hardware dispatch.

The outer Wago call time includes overhead and lock waiting not included in `Last.Total`. Raw snapshot fields are Go `time.Duration` values. The demonstration's JSON durations are integer nanoseconds.

### 25.2 Proposed extension diagnostics

**Proposed.** Preserve the existing snapshot fields and their legacy interpretation. Add `func (p *Plugin) BufferSnapshot() BufferSnapshot`. It returns an owned value copy, including copies of slices, under the plugin lock. It does not expose mutable live state. Diagnostic history MUST remain bounded; a last-operation record and aggregate counters are sufficient.

```go
type BufferSnapshot struct {
    PeakRuntimeBufferBytes      uint64
    State, Reason               string
    DeviceState                 string
    Last                        BufferOperation
    Totals                      TransferCounters
    TotalsComplete              bool
    Kernels                     []KernelDiagnostic
    Instances                   []InstanceDiagnostic
    PipelineHits, PipelineMisses uint64
    PendingCompilations         uint32
    PendingContracts, PendingTranslations uint32
    PendingContractBytes, PendingWGSLBytes uint64
    ActiveCompilations          uint32
    RuntimeBufferBytes          uint64
    RuntimePoolBytes, LegacyBufferBytes uint64
}

type BufferOperation struct {
    Sequence                    uint64
    Instance                    wago.InstanceIdentity
    KernelID                    uint32 // Zero if there is no selected kernel.
    Import                      string
    Status                      int32
    ReasonCode, Reason, Outcome  string
    Count                       uint32
    HardwareSubmitted           bool
    TransferCountsComplete      bool
    ProfileValid, TimestampsValid bool
    QueueWait, Total            time.Duration
    Upload, DeviceCopy, Compute, Download time.Duration
    Conversion, Commit, DeviceCompute time.Duration
    TransferCounters
}

type TransferCounters struct {
    GuestSetBytes, GuestCopyBytes             uint64
    GuestSetCount, GuestCopyCount             uint64
    LogicalUploadBytes, LogicalDownloadBytes  uint64
    GPUUploadBytes, GPUDownloadBytes          uint64
    UploadCount, DownloadCount                uint64
    DeviceCopyBytes, DeviceCopyCount          uint64
    ParameterUploadBytes, ParameterUploadCount uint64
}

type KernelDiagnostic struct {
    Translate, Pipeline time.Duration
    Module           wago.ModuleIdentity
    KernelID         uint32
    State, Reason    string
    ContractVerified bool
}

type InstanceDiagnostic struct {
    PeakBytes uint64
    Instance wago.InstanceIdentity
    Buffers, CPUCurrent, GPUCurrent, BothCurrent, ContentsLost uint32
    LogicalBytes, CPUBytes, GPUBytes, StagingBytes, ScratchBytes uint64
    RetainedPoolBytes, ReservedBytes uint64
}
```

Identity fields use Wago's public identity types. They are diagnostic identifiers, not guest handles. All durations use Go `time.Duration`.

Runtime state strings are `DISABLED`, `UNAVAILABLE`, `READY`, `FAILED`, `STOPPING`, and `STOPPED`. `DeviceState` uses section 9.5's states, or `UNAVAILABLE` before creation. Kernel states are `UNVERIFIED_CONTRACT`, `INVALID_CONTRACT`, `CPU_ONLY`, `PIPELINE_READY`, and `PIPELINE_FAILED`.

`ReasonCode` is `NONE`, `DISABLED`, `NO_DEVICE`, `FLOAT_MODE`, `BELOW_THRESHOLD`, `UNSUPPORTED_KERNEL`, `PIPELINE_FAILURE`, `SOURCE_MISMATCH`, `CONTRACT_UNVERIFIED`, `RESOURCE_LIMIT`, `DEVICE_FAILURE`, `CONTENTS_LOST`, `CANCELLED`, or `CONTRACT_ERROR`. It identifies the decision. Numeric status controls guest behavior. Text is descriptive, not a stable machine interface. Do not include guest data or raw process addresses.

#### Submission and confirmed execution

Before any recorded operation, `Sequence` is zero and `Last` has zero values. Later sequence values increase without wrapping. Dispatch outcomes are `GPU`, `CPU_FALLBACK_READY`, `NOOP`, or `ERROR`. Management and transfer success use `OK`. Fallback-ready does not prove the guest subsequently finished its CPU loop.

`HardwareSubmitted` means the backend submitted commands to a hardware adapter. It does **not** prove that a kernel ran. A failed operation can set it, including a rejected command later treated as a no-op. Confirmed successful GPU execution requires a nonzero dispatch, a hardware adapter, status `OK`, outcome `GPU`, and every completion/error/commit check in section 23.4. Successful queue completion alone is insufficient. There is no `HardwareRan` field.

An outer Wasm cancellation or trap does not rewrite the committed import's diagnostic record. The host must interpret the import record and outer result separately.

#### Transfer boundaries and totals

| Boundary | Byte/count fields | Exact meaning |
| --- | --- | --- |
| Guest → plugin CPU | `GuestSetBytes`, `GuestSetCount` | Canonical payload copied by a committed nonempty setBuffer operation |
| Plugin CPU → guest | `GuestCopyBytes`, `GuestCopyCount` | Canonical payload copied by a committed nonempty copyBuffer operation |
| Plugin CPU → GPU | `LogicalUploadBytes`, `GPUUploadBytes`, `UploadCount` | Canonical equivalent bytes, physical payload bytes, and completed nonempty payload transfer operations |
| GPU → plugin CPU | `LogicalDownloadBytes`, `GPUDownloadBytes`, `DownloadCount` | Canonical equivalent bytes, physical payload bytes, and completed nonempty payload readback operations |
| GPU → GPU | `DeviceCopyBytes`, `DeviceCopyCount` | Physical bytes and completed nonempty device-local copies, including temporary-output seeds and growth copies |
| Dispatch parameters → GPU | `ParameterUploadBytes`, `ParameterUploadCount` | Actual completed parameter-block transfers, separate from payload uploads |

These counters measure explicit application transfers, not bus traffic, driver-internal copies, or shader memory accesses. Each transfer is counted once at its boundary. A guest set followed by a GPU upload appears in two distinct categories, not twice in upload. A seed copy never appears in upload or download. Full-buffer coherence costs are counted in full even when logical work covers one element. Zero-count calls count no transfers.

`Last` contains this operation's known completed transfers. Transfers can have completed even when its later commit fails. Unknown partial native work MUST NOT be invented as a byte count. `TransferCountsComplete=false` marks any unresolved transfer; reported counters are then lower bounds. When true, all relevant transfer results are known, including zero for stages not run.

`Totals` accumulates the same boundaries for all v1 imports during the runtime lifetime, including implicit readback from scalar access. `Last` records only createBuffer, createBufferPacked, freeBuffer, bindBuffer, dispatch, and the six bulk-transfer imports. getBuffer, readBuffer*, and writeBuffer* MUST NOT overwrite `Last` or advance its sequence, including when they trap. Scalar traps use host-trap diagnostics. These operations do not allocate a history record. An implicit transfer does update `Totals`. Thus a CPU loop preserves the last dispatch record while transfer accounting remains complete. Legacy transfer counters retain their existing snapshot interface and are excluded from these v1 totals. A mixed-ABI report combines them explicitly. `TotalsComplete` starts true and becomes false after any unknown transfer or counter saturation. Counters saturate rather than wrap. No unbounded trace is required.

#### Timing validity

`QueueWait` and `Total` are always valid for a recorded operation (`Sequence != 0`). `Total` starts at import entry and ends at its terminal decision/copy; it includes queue wait, validation, and commit. It is not the enclosing Wasm invocation time. `QueueWait` measures time waiting for serialized backend admission.

`ProfileValid` is **all-or-nothing** for `Upload`, `DeviceCopy`, `Compute`, `Download`, `Conversion`, and `Commit`. It is true only when stage profiling is enabled and the operation completes successfully, including a prepared fallback, with all stages that ran measured. Stages that did not run are valid zero durations. On an error, disabled profiling, or any incomplete stage measurement, it is false and all six optional fields MUST be zero. `QueueWait` and `Total` remain usable on failure. This policy intentionally discards partial stage timing to avoid ambiguous fields.

The `DeviceCopy` stage measures encoding, submission, and waiting for device-local copies when profiling separates them. It is host wall time, not a hardware timestamp. Extra profiling waits can change performance. `DeviceCompute` has independent `TimestampsValid`; when false, it MUST be zero and treated as unavailable. Stage profiling does not imply timestamp support.

For chain measurements, add guest set/copy operations, dispatches, and any fallback. Report resident dispatch time separately. Do not present it as end-to-end time.

#### Retention and resource counters

`PendingContracts` and `PendingContractBytes` count retained unattached CPU records and their accounted storage. `PendingTranslations` and `PendingWGSLBytes` count retained unattached translations and text. `PendingCompilations` is the count of the union of their compilation identities, not their sum. Missing/unverified identities are not retained just to count them. `ActiveCompilations` is zero or one under section 12.1's permit. These fields make pending-table saturation distinct from GPU pipeline failure.

`LogicalBytes` sums live buffer lengths times canonical widths. CPU, GPU, staging, scratch, and retained-pool allocation categories are disjoint; their sum is `ReservedBytes`. `RuntimeBufferBytes` sums instance reservations plus shared runtime-pool and legacy-buffer bytes, without double counting. Retired resources keep their charge until release. Unknown native overhead is excluded, not reported as a measured zero cost.

Return slices in stable order for a given set of identities and IDs. Snapshot construction may allocate bounded copies; it is outside the timed execution path.

## 26. Build, distribution, and runnable artifacts

**Current dependencies.** `go.mod` declares Go 1.25 and pins:

| Dependency | Version |
| --- | --- |
| Wago | `v0.1.0-beta.12.0.20261007220511-7aa401f29a33` |
| oliverbestmann WebGPU | `v1.36.0`, local safety repair |
| `golang.org/x/sys` | `v0.30.0`, indirect |

The tested toolchain is Go 1.27.1 on Linux/amd64. The module's declared Go version is not evidence that every dependency and backend was tested with that toolchain minimum. Pin changes require the relevant CPU and hardware checks.

Run these existing commands from the project directory:

```sh
# Portable plugin tests and CPU fallback examples.
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 go run ./cmd/demo -cpu-only
CGO_ENABLED=0 go run ./examples/start -cpu-only

# GPU examples. Build the pinned native patch first; Rust/Cargo is required.
./native/build.sh
go run -tags webgpu ./cmd/demo -require-gpu
go run -tags webgpu ./examples/start -require-gpu

# Explicit real-hardware tests; absence of hardware must fail this mode.
WAGO_GPU_TEST=1 go test -tags webgpu ./...
go test -race ./...
WAGO_GPU_TEST=1 go test -race -tags webgpu ./...

# Rebuild the existing checked-in fixtures with WABT.
go generate ./internal/fixtures
go generate ./examples/start

# Existing Go benchmarks with allocation reporting.
GOMAXPROCS=1 WAGO_GPU_TEST=1 go test -tags webgpu -run '^$' \
  -bench 'Benchmark(Batch)?(CPU|GPU)$' -benchtime=50x -benchmem
```

No GPU device or C compiler is required for the `CGO_ENABLED=0` test path. WABT is needed to regenerate WAT fixtures, not to run tests against checked-in Wasm. A GPU-tagged build with CGO disabled still selects the stub; it is not a hardware test.

The current demo provides `-cpu-only`, `-require-gpu`, `-kernel`, `-n`, `-sizes`, `-passes`, `-min-elements`, `-bench`, `-reps`, `-profile`, `-separate`, and `-json`. `-require-gpu` MUST fail if required work uses fallback. `-separate` is a transfer-cost control, not a chain-rollback API. The square fixture's demo caps passes at four to keep its inputs in the intended numerical range; the plugin batch limit remains 64.

**Runnable artifacts.** [examples/buffers](examples/buffers/main.go) supplies the Go host, full WAT, and Wasm for section 13, with CPU and require-hardware modes. [examples/tinygo](examples/tinygo/main.go) runs the complete guest-language path. [examples/storage](examples/storage/main.go) provides F16 CPU/GPU execution plus memory64 and GC transfer examples. Proposed snippets alone are not runnable deliverables.

The library MUST NOT install a driver, alter Wago source, or download executable code at runtime. Normal Go module and native-library build dependencies remain build-time concerns. The project SHOULD avoid adding C or C++ source and SHOULD keep the backend behind the existing small private interface.

## 27. Compatibility, exclusions, and evolution

Once `wago_gpu_v1` is frozen, the following are fixed: import names and signatures, type IDs, status values, index units, transfer byte order, handle isolation, F16 conversion, commit rules, and fallback meaning. Changing one of these requires a new import-module version or an explicitly compatible addition.

New optional imports can be added without changing old signatures. An older plugin will fail to link a module that requires an unavailable import; it MUST NOT substitute an import with similar spelling. No guest feature-query import is defined in version 1. The host selects compatible modules and configuration.

Legacy modules continue to use `wago_gpu` and their existing status table. A module can import both interfaces. A legacy call does not read plugin buffer handles, and a buffer dispatch does not automatically read memory 0. The guest must make the needed copies explicitly.

Buffer data, handles, pipelines, and devices are runtime resources. This specification does not define snapshot serialization, checkpoint restoration of plugin state, cross-process sharing, or transfer of a compiled GPU pipeline to another runtime. A host that restores Wasm memory must recreate buffer state explicitly.

The following remain outside version 1:

- Automatic replacement of ordinary Wasm functions or a general shader JIT for arbitrary Wasm.
- Shader access to guest linear memory, native pointers, GC objects, or reference arrays.
- Texture, sampler, graphics, vertex, fragment, and rendering APIs.
- Reductions, neighbor communication, arbitrary loops, indirect calls, recursion, exceptions, SIMD lowering, atomics, and synchronization within a kernel.
- Asynchronous guest dispatch handles, futures, event callbacks, and a public command-buffer API.
- Multi-GPU scheduling, device hot replacement, automatic replay, or whole-chain rollback.
- Automatic numerical equivalence for all Wasm float inputs, native 64-bit WGSL arithmetic, or implicit type conversion between buffers.
- A promise that Go context cancellation can interrupt a hung native driver.

These exclusions do not prevent ordinary CPU Wasm code from using features that Wago supports. They only limit the plugin's GPU path and buffer-transfer contract.

## 28. Conformance and release checklist

Complete v1 requires all 34 CPU imports and the declared GPU subsets. A release report MUST identify the exact source revision, the metadata profile and guest compiler, and distinguish these milestones. A partial implementation MUST NOT claim the full version 1 contract.

| ID | Required outcome | Evidence |
| --- | --- | --- |
| P01 | External optional Go plugin; no GPU-specific Wago modification | Dependency pin, source review, and documented/tested generic dependency changes |
| P02 | Correct scalar translation from actual instructions | `TestTranslation`, malformed-input tests, and compiler fuzzing |
| P03 | Legacy behavior remains stable | Scalar, batch, overlap, range, and fallback tests |
| P04 | Buffer ABI and Go configuration match this document | Import-signature tests, configuration validation, complete example |
| P05 | Exact caller, module, and source association | Cross-instance, changed-source, concurrent-compilation, and cache tests |
| P06 | Checked guest storage with no retained borrow | Memory32, memory64, GC, overflow, and callback-lifetime tests |
| P07 | Correct typed access and F16 conversions | Width tests, exhaustive F16 reads, boundary writes, hardware conversion tests |
| P08 | Real multi-kernel GPU execution | Require-hardware chain run with verified outputs and transfer counts |
| P09 | Current data and safe fallback | GPU-to-CPU transition, failed-commit, device-loss injection, and lost-content tests |
| P10 | Bounded memory and complete cleanup | Budget, reuse, start-trap, live-instance module-close, and shutdown tests |
| P11 | Supported concurrency has no shared-data failure | Separate-instance stress and Go race detector |
| P12 | Measured CPU and GPU performance | Four required sizes, both CPU baselines, first/repeated times, memory and hardware metadata |
| P13 | Honest optional measurements and compatibility | Timestamp-validity tests, unavailable-device mode, legacy ABI tests |

Existing named tests are evidence only for the code they exercise. For example, current `TestHardware` does not prove the proposed handle API. Proposed tests can use equivalent names; their required behavior is the contract.

Compiler fuzzing MUST include truncated sections, invalid LEB encodings, wrong types, unsupported calls, and bounded source sizes. Host-import tests MUST check that an error does not change a destination. Lifecycle tests MUST check retained resource counts after the relevant close operation, not only the absence of a crash.

A release that claims hardware support MUST run the primary examples on a hardware adapter. It MUST keep the generated WGSL or an inspectable translation result so reviewers can verify that the shader came from the selected body. CPU-only test success is required but is not GPU evidence.

## 29. Existing evidence and remaining work

**Recorded evidence, not a new run.** The existing scalar experiment ran on an NVIDIA RTX 4060 Laptop GPU with NVIDIA driver 550.163.01, Debian 13.7, Linux 6.12.111, Go 1.27.1, and an AMD Ryzen 7 8845HS CPU. The retained backend was Vulkan through cogentcore WebGPU. The original `2*x` and `x*x+1` kernels were translated from actual Wasm and compared with Wago CPU execution.

The later single-pass results below are repeated-call medians in milliseconds. They include GPU transfers and result commit. They are copied from [IMPROVEMENTS.md](IMPROVEMENTS.md) and [its saved measurements](results/improvements-single.json), not measurements of the proposed buffer API.

| Kernel | Elements | Native Wago CPU | GPU total |
| --- | ---: | ---: | ---: |
| `2*x` | 1,024 | 0.0020 | 0.0511 |
| `2*x` | 16,384 | 0.0213 | 0.0695 |
| `2*x` | 1,000,000 | 1.3050 | 0.9855 |
| `2*x` | 10,000,000 | 14.9880 | 11.1657 |
| `x*x+1` | 1,024 | 0.0020 | 0.0437 |
| `x*x+1` | 16,384 | 0.0211 | 0.0589 |
| `x*x+1` | 1,000,000 | 1.3025 | 0.9288 |
| `x*x+1` | 10,000,000 | 15.0646 | 10.8100 |

Small workloads were slower on the GPU. First-call costs, device creation, stage measurements, allocation data, and heavier/batched workloads are in the linked report. These medians do not establish a general speed guarantee or the crossover point for buffers with per-element CPU host calls.

Recorded CPU, GPU, race, repeated-execution, cleanup, fallback, and fuzz tests passed for the existing interface. Direct device timestamps, a usable device-loss callback, physical device-removal recovery, and a hard driver-hang timeout were not established. The later source review confirmed callback-context lifetime and error-reporting gaps in the retained binding. Those gaps were not disproved by the recorded arithmetic tests. Only the stated hardware platform has evidence here.

The named buffer kernels, typed handles, persistent buffers, F16 conversions, memory64 transfers, and GC transfers are now implemented. Current test and hardware evidence is in [BUFFER_REPORT.md](BUFFER_REPORT.md). No Wago source change was needed. The earlier measurements above remain historical scalar results.

## 30. Specification validation

The complete proposed WAT module in section 13 MUST pass `wat2wasm` and `wasm-validate` for syntax and Wasm type correctness. This check does not execute its host imports and does not prove GPU behavior. Runnable implementation tests remain separate.

Document maintenance MUST check local links, import signatures, result order, unique type and status IDs, and agreement between the host manifest and the WAT example. Run `python3 spec/check_spec.py` from the project root with Python 3, Go/gofmt, WABT, and wasm-tools installed. The all-import declaration check uses wasm-tools for `anyref`; the installed WABT does not parse that GC type. The non-GC complete example still uses WABT. It checks every proposed ABI signature in a generated WAT import module, the example's imported subset, type/status tables, packed-result bit layout, local links, Go snippet syntax, and TinyGo source import declarations. It writes [spec/validation.json](spec/validation.json) with the exact scope and file hashes. `--tinygo` also recompiles and inspects the guest probe.

The generated registration table is now checked against the canonical ABI by `TestCanonicalABI`. Runtime tests, the complete TinyGo guest, and hardware evidence are separate from this document checker. The implementation report lists their results and remaining limits. A generated import-only module is a declaration check, not a functioning guest or a hardware test.

 When an interface is implemented, update its status and add the corresponding test evidence. Do not silently replace historical benchmark values with new runs.


## 31. Review decisions and release gates

This revision incorporates both supplied reviews and checks the source-identity and cancellation claims against the pinned local source. The second review covered the complete document through section 31. The document map covers every numbered section through this section.

| Review area | Decision and location | Evidence still required |
| --- | --- | --- |
| Final CPU metadata | Separate contract/WGSL/pipeline records; restricted known-source profile; generic metadata API proposal, sections 14 and 22 | Mismatch, eviction, shared memory, precompiled-module tests |
| Native callbacks | Confirmed retained unpinned addresses; repair/replacement is mandatory, section 23.5 | Delayed native callbacks and safe retirement under GC, timeout, and shutdown |
| Completion/error reporting | Await successful work and completed error checks, section 23.4 | Delayed error after successful completion |
| Exact F16 | Integer-encoded exact values until arithmetic, section 6.1 | Generated helpers and exhaustive conversion-only hardware tests |
| Cancellation/device health | One commit decision; no quarantine recovery; cross-instance/cross-ABI invalidation, section 9 | Commit-race and late-callback tests |
| Stop order | No wait for terminal close observers; independent ownership, section 21 | Blocked-dispatch shutdown test |
| Import provider | Host MUST NOT override GPU intrinsics, section 21 | Controlled-host override rejection |
| Compiler tracking | Immutable computed values, value categories, expanded bounds, exact phase table, section 7 | Mutation, alias/category, and decoder tests |
| Error classification | Invalid contract differs from unsupported lowering, section 7.3 | Stable status and no accidental fallback tests |
| Guest compiler | TinyGo 0.42.0 selected; packed companion; compiled probe, sections 5 and 20 | Full management guest on CPU and GPU before ABI freeze |
| Immutable GC | Reserve complete hidden copy before GCArrayBytes, section 5 | Small transfer from large immutable array under budget |
| Allocation ownership | Atomic runtime reservations include legacy, pools, retirement and growth, section 12 | Actual peak measurements and fallback preparation |
| Bulk commit | Prepare then non-failing final commit, section 24 | Failure/cancellation boundary tests |
| Complete-document review | Both reviews incorporated; sections 25–30 retained and clarified | Re-run cross-file checks after later ABI edits |

### 31.1 Second-review decisions

| Finding | Decision | Required interaction test |
| --- | --- | --- |
| Prepared artifact identity | Only Runtime.Compile and PreparedCompile.Compile; controlled host rejects Adopt | Prepare A, attempt artifact B with equal signatures/different arithmetic |
| Pending CPU metadata | Oldest unattached record is evicted; no unbounded rejected-ID set | Close more than 64 preparations, then verify a fresh module |
| Established contract errors | Durable contract owns them, never WGSL alone | Evict WGSL after READ/type/slot violations, including GPU-disabled mode |
| Permission versus output | Actual read/write sets; only actual stores create outputs | Read-only READ_WRITE, unused WRITE, identical-value store, multiple outputs |
| Import versus outer result | Outer cancellation/trap does not roll back import commit | Real Wago boundary with fake backend and post-commit cancellation |
| Integer data categories | Constants and original index are valid store data; saved-index restoration is valid | Constant/index fill, sign-bit patterns, narrowing, restore, reject handle data |
| Diagnostic proof | Submitted differs from confirmed commit; separate transfer boundaries | Rejected command cannot report GPU success; small-prefix traffic accounting |
| Timing validity | Optional stages valid together only on profiled success | Fail between stages and verify unavailable fields are zero |
| Active compiler work | One private permit per runtime; preserve verified metadata on lowering limits | Concurrent hooks, bounded workspace, permit release on failure |
| Zero count and timeout | Validation/limits then no-op before cancellation; scalar readback has timeout | Cancelled no-op and scalar implicit-readback timeout |
| Host cleanup | Independent context and joined close errors | Cancel work or inject Stop error; cleanup still starts and error is returned |
| Exact float records | Arithmetic materialization leaves saved exact bits intact | One F32 read feeds relaxed arithmetic and an exact-copy output |

The source checks found a **plugin-view** metadata gap, not a total absence of public host metadata APIs. The original binding had a device-loss callback that the original backend did not register. The repaired backend now registers it and consumes loss notifications under the plugin lock. The asynchronous context defect was confirmed by source inspection; no crash was reproduced. Historical successful GPU results remain recorded observations, not proof of callback safety or complete error reporting.

The earlier document review did not accept a replacement binding or execute buffers. The implementation now uses a local copy of oliverbestmann/webgpu v1.36.0 plus a pinned native error-reporting patch; see [dependency patch notes](third_party/webgpu/PATCHES.md). Wago remains unchanged. The compile-only TinyGo probe and WAT validation are stated separately from runtime tests.

Do not freeze the import contract until the full guest-language path passes. Do not accept the GPU buffer phase until callback ownership and the completed error protocol pass. An implementation agent can begin the bounded CPU and fake-backend work under this draft; it cannot label an untested backend or missing metadata path conforming.

The third review adds five local clarifications: exact Last-recording imports, public CompileError categories, alias checks over all declared slots, read-before-write seed requirements, and adoption restrictions for CPU metadata as well as GPU use. These requirements apply to implementation and tests B21 and B33–B36.


### 31.2 Implementation evidence

The local implementation uses all 34 canonical imports. `NewHost` prevents
artifact adoption and intrinsic overrides by construction. The lower-level
PluginSet still requires the host obligations in section 22.1.1.

The native build is restricted to Linux with the patched dependency. Its error
sink has validation, allocation, and device-loss categories. The plugin checks
all three supported paths; it does not request the unsupported Internal scope.
Submission, polling, and mapped-range errors use the patched native error sink.
A real test rejects a resource destroyed after command encoding, then receives
a successful queue notification without reporting the failed dispatch as a
successful execution.

`KernelDiagnostic.Translate` and `Pipeline` report build wall times.
`PeakRuntimeBufferBytes` and `InstanceDiagnostic.PeakBytes` report tracked
reservation peaks. They exclude unknown driver overhead, as do the other budget
fields. The benchmark records process RSS separately.

Current evidence, commands, numerical results, and untested hardware conditions
are in [BUFFER_REPORT.md](BUFFER_REPORT.md). The ABI is not a published release.
