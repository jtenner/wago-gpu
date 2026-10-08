(module
  ;; Arguments: input byte offset, output byte offset, element count.
  ;; Result: 0 = GPU success, 1 = use CPU, 2 = invalid memory range.
  (import "wago_gpu" "run"
    (func $gpu (param i32 i32 i32) (result i32)))

  (memory (export "memory") 1)
  (global $used_gpu (mut i32) (i32.const 0))

  ;; The host selects this export. The plugin compiles its body into WGSL.
  (func $kernel (export "kernel") (param $x f32) (result f32)
    (f32.mul (local.get $x) (f32.const 2)))

  ;; CPU fallback for this example's four values.
  ;; Read bytes 0..15 and write bytes 16..31.
  (func $cpu
    (local $offset i32)
    (loop $next
      (f32.store
        (i32.add (i32.const 16) (local.get $offset))
        (call $kernel (f32.load (local.get $offset))))
      (local.set $offset
        (i32.add (local.get $offset) (i32.const 4)))
      (br_if $next (i32.lt_u (local.get $offset) (i32.const 16)))))

  ;; This function runs automatically when Wago creates the instance.
  (func $start
    (local $status i32)

    ;; Input array: [1, 2, 3, 4]. Each f32 uses four bytes.
    (f32.store (i32.const 0)  (f32.const 1))
    (f32.store (i32.const 4)  (f32.const 2))
    (f32.store (i32.const 8)  (f32.const 3))
    (f32.store (i32.const 12) (f32.const 4))

    ;; Apply the selected kernel to all four values on the GPU.
    (local.set $status
      (call $gpu (i32.const 0) (i32.const 16) (i32.const 4)))

    (if (i32.eq (local.get $status) (i32.const 1))
      (then (call $cpu))
      (else
        ;; Any status other than success stops instantiation.
        (if (i32.ne (local.get $status) (i32.const 0))
          (then unreachable))))

    (global.set $used_gpu
      (i32.eq (local.get $status) (i32.const 0))))

  (start $start)

  ;; The host reads this flag after instantiation. It starts no GPU work.
  (func (export "used_gpu") (result i32)
    (global.get $used_gpu))
)
