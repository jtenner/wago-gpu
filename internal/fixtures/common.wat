;; Native Wago CPU reference. No GPU imports or compiler extension.
;; A, B, C, transposed B occupy consecutive count*4 byte regions.
(module
 (memory (export "memory") 1)
 (func (export "reserve") (param i32) (result i32) (memory.grow (local.get 0)))
 (func (export "transpose") (param $n i32) (local $i i32) (local $m i32)
  (local.set $m (i32.mul (local.get $n) (local.get $n)))
  (block $done (loop $next
   (br_if $done (i32.ge_u (local.get $i) (local.get $m)))
   (f32.store
    (i32.mul (i32.add (i32.mul (local.get $m) (i32.const 3))
     (i32.add (i32.mul (i32.rem_u (local.get $i) (local.get $n)) (local.get $n))
      (i32.div_u (local.get $i) (local.get $n)))) (i32.const 4))
    (f32.load (i32.mul (i32.add (local.get $m) (local.get $i)) (i32.const 4))))
   (local.set $i (i32.add (local.get $i) (i32.const 1))) (br $next))))
 (func $matmul (param $n i32) (param $transposed i32)
  (local $i i32) (local $k i32) (local $m i32) (local $row i32) (local $col i32) (local $b i32) (local $sum f32)
  (local.set $m (i32.mul (local.get $n) (local.get $n)))
  (block $done (loop $next
   (br_if $done (i32.ge_u (local.get $i) (local.get $m)))
   (local.set $row (i32.mul (i32.div_u (local.get $i) (local.get $n)) (local.get $n)))
   (local.set $col (i32.rem_u (local.get $i) (local.get $n)))
   (local.set $k (i32.const 0)) (local.set $sum (f32.const 0))
   (block $inner_done (loop $inner
    (br_if $inner_done (i32.ge_u (local.get $k) (local.get $n)))
    (local.set $b (if (result i32) (local.get $transposed)
     (then (i32.add (i32.mul (local.get $m) (i32.const 3))
      (i32.add (i32.mul (local.get $col) (local.get $n)) (local.get $k))))
     (else (i32.add (local.get $m) (i32.add (i32.mul (local.get $k) (local.get $n)) (local.get $col))))))
    (local.set $sum (f32.add (local.get $sum) (f32.mul
     (f32.load (i32.mul (i32.add (local.get $row) (local.get $k)) (i32.const 4)))
     (f32.load (i32.mul (local.get $b) (i32.const 4))))))
    (local.set $k (i32.add (local.get $k) (i32.const 1))) (br $inner)))
   (f32.store (i32.mul (i32.add (i32.mul (local.get $m) (i32.const 2)) (local.get $i)) (i32.const 4)) (local.get $sum))
   (local.set $i (i32.add (local.get $i) (i32.const 1))) (br $next))))
 (func (export "matmul") (param i32) (call $matmul (local.get 0) (i32.const 0)))
 (func (export "matmul_t") (param i32) (call $matmul (local.get 0) (i32.const 1)))
 (func (export "saxpy") (param $n i32) (local $i i32)
  (block $done (loop $next
   (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
   (f32.store (i32.mul (i32.add (i32.mul (local.get $n) (i32.const 2)) (local.get $i)) (i32.const 4))
    (f32.add (f32.mul (f32.const 2) (f32.load (i32.mul (local.get $i) (i32.const 4))))
     (f32.load (i32.mul (i32.add (local.get $n) (local.get $i)) (i32.const 4)))))
   (local.set $i (i32.add (local.get $i) (i32.const 1))) (br $next))))
 (func (export "dot") (param $n i32) (result f32) (local $i i32) (local $sum f32)
  (block $done (loop $next
   (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
   (local.set $sum (f32.add (local.get $sum) (f32.mul
    (f32.load (i32.mul (local.get $i) (i32.const 4)))
    (f32.load (i32.mul (i32.add (local.get $n) (local.get $i)) (i32.const 4))))))
   (local.set $i (i32.add (local.get $i) (i32.const 1))) (br $next)))
  (local.get $sum)))
