(module
  (import "wago_gpu" "run" (func $gpu (param i32 i32 i32) (result i32)))
  (import "wago_gpu" "run_batch" (func $gpu_batch (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1 2048)
  (func (export "reserve") (param i32) (result i32)
    local.get 0 memory.grow)
  (func $kernel (export "kernel") (param f32) (result f32)
    local.get 0
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add
    f32.const 0.9990234375
    f32.mul
    f32.const 0.0009765625
    f32.add)
  (func $cpu (export "cpu") (param $input i32) (param $output i32) (param $count i32)
    (local $i i32)
    block $done
      loop $next
        local.get $i local.get $count i32.ge_u br_if $done
        local.get $output local.get $i i32.const 4 i32.mul i32.add
        local.get $input local.get $i i32.const 4 i32.mul i32.add
        f32.load call $kernel f32.store
        local.get $i i32.const 1 i32.add local.set $i
        br $next
      end
    end)
  (func (export "gpu") (param i32 i32 i32) (result i32)
    local.get 0 local.get 1 local.get 2 call $gpu)
  (func (export "run") (param i32 i32 i32) (result i32)
    (local $status i32)
    local.get 0 local.get 1 local.get 2 call $gpu local.set $status
    local.get $status i32.const 1 i32.eq
    if
      local.get 0 local.get 1 local.get 2 call $cpu
    end
    local.get $status)

  (func $cpu_batch (export "cpu_batch") (param $input i32) (param $output i32) (param $count i32) (param $passes i32)
    (local $j i32)
    block $done
      loop $next
        local.get $j local.get $passes i32.ge_u br_if $done
        local.get $input local.get $output local.get $count call $cpu
        local.get $output local.set $input
        local.get $j i32.const 1 i32.add local.set $j
        br $next
      end
    end)
  (func (export "gpu_batch") (param i32 i32 i32 i32) (result i32)
    local.get 0 local.get 1 local.get 2 local.get 3 call $gpu_batch)
  (func (export "run_batch") (param i32 i32 i32 i32) (result i32)
    (local $status i32)
    local.get 0 local.get 1 local.get 2 local.get 3 call $gpu_batch local.set $status
    local.get $status i32.const 1 i32.eq
    if
      local.get 0 local.get 1 local.get 2 local.get 3 call $cpu_batch
    end
    local.get $status)
  ;; Benchmark control: one guest call with a full transfer per pass.
  ;; Abort on GPU failure; use run_batch for transactional CPU fallback.
  (func (export "gpu_separate") (param $input i32) (param $output i32) (param $count i32) (param $passes i32) (result i32)
    (local $j i32) (local $status i32)
    local.get $passes i32.eqz if i32.const 3 return end
    local.get $passes i32.const 64 i32.gt_u if i32.const 3 return end
    block $done
      loop $next
        local.get $j local.get $passes i32.ge_u br_if $done
        local.get $input local.get $output local.get $count call $gpu local.tee $status
        if local.get $status return end
        local.get $output local.set $input
        local.get $j i32.const 1 i32.add local.set $j
        br $next
      end
    end
    i32.const 0)
)
