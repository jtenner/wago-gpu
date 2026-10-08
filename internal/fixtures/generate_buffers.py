"""Generate the benchmark guest. The plugin compiles the resulting Wasm code."""
from pathlib import Path
parts = ['''(module
(import "wago_gpu_v1" "createBufferPacked" (func $create (param i32 i32) (result i64)))
(import "wago_gpu_v1" "freeBuffer" (func $free (param i32) (result i32)))
(import "wago_gpu_v1" "bindBuffer" (func $bind (param i32 i32) (result i32)))
(import "wago_gpu_v1" "getBuffer" (func $get (param i32) (result i32)))
(import "wago_gpu_v1" "readBufferF32" (func $read (param i32 i32) (result f32)))
(import "wago_gpu_v1" "writeBufferF32" (func $write (param i32 i32 f32)))
(import "wago_gpu_v1" "dispatch" (func $dispatch (param i32 i32) (result i32)))
(import "wago_gpu_v1" "setBuffer32" (func $set (param i32 i32 i32 i32 i32) (result i32)))
(import "wago_gpu_v1" "copyBuffer32" (func $copy (param i32 i32 i32 i32 i32) (result i32)))
(memory (export "memory") 1)
(global $input (mut i32) (i32.const 0))
(global $output (mut i32) (i32.const 0))
(global $count (mut i32) (i32.const 0))
(func $check (param i32) (if (local.get 0) (then unreachable)))
(func (export "reserve") (param i32) (result i32) (memory.grow (local.get 0)))
(func (export "setup") (param $n i32) (local $packed i64)
 (global.set $count (local.get $n))
 (local.set $packed (call $create (i32.const 8) (local.get $n)))
 (call $check (i32.wrap_i64 (i64.shr_u (local.get $packed) (i64.const 32))))
 (global.set $input (i32.wrap_i64 (local.get $packed)))
 (local.set $packed (call $create (i32.const 8) (local.get $n)))
 (call $check (i32.wrap_i64 (i64.shr_u (local.get $packed) (i64.const 32))))
 (global.set $output (i32.wrap_i64 (local.get $packed)))
 (call $check (call $bind (i32.const 0) (global.get $input)))
 (call $check (call $bind (i32.const 1) (global.get $output))))
(func (export "upload") (result i32)
 (call $set (global.get $input) (i32.const 0) (i32.const 0) (i32.const 0) (global.get $count)))
(func (export "download") (result i32)
 (call $copy (global.get $output) (i32.const 0) (i32.const 0) (i32.mul (global.get $count) (i32.const 8)) (global.get $count)))
(export "dispatch" (func $dispatch))
''']
for name, number in [('twice',1),('square',2)]:
    load='(call $read (call $get (i32.const 0)) (local.get $i))'
    expr=f'(f32.mul {load} (f32.const 2))' if name=='twice' else f'(f32.add (f32.mul {load} {load}) (f32.const 1))'
    parts.append(f'''(func ${name} (export "wago_gpu.kernel.{name}") (param $i i32)
 (call $write (call $get (i32.const 1)) (local.get $i) {expr}))
(func (export "wago_gpu.cpu.{name}") (param $n i32) (local $i i32)
 (block $done (loop $next
 (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
 (call ${name} (local.get $i))
 (local.set $i (i32.add (local.get $i) (i32.const 1))) (br $next))))
''')
    load='(f32.load (i32.mul (local.get $i) (i32.const 4)))'
    expr=f'(f32.mul {load} (f32.const 2))' if name=='twice' else f'(f32.add (f32.mul {load} {load}) (f32.const 1))'
    parts.append(f'''(func (export "direct.{name}") (param $n i32) (local $i i32)
 (block $done (loop $next
 (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
 (f32.store (i32.mul (i32.add (local.get $n) (local.get $i)) (i32.const 4)) {expr})
 (local.set $i (i32.add (local.get $i) (i32.const 1))) (br $next))))
''')
parts.append(')\n')
Path(__file__).with_name('buffer_work.wat').write_text(''.join(parts))
