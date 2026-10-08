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
