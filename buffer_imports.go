package wagogpu

import (
	"context"
	"encoding/binary"
	"fmt"
	wago "github.com/wago-org/wago"
	"math"
	"strings"
	"time"
)

func (p *Plugin) registerBuffers(reg *wago.Registrar) error {
	imports, e := reg.HostImports()
	if e != nil {
		return e
	}
	for _, entry := range bufferImports {
		entry := entry
		imports.HostFunc("wago_gpu_v1", entry.name, func(c wago.Caller, call wago.HostCall) { p.bufferCall(c, call, entry.name) }).Params(entry.params...).Results(entry.results...).Docs("Provisional v1 plugin-owned buffer API")
	}
	return nil
}
func bufferTrap(status int32) {
	panic(wago.HostTrap{Err: fmt.Errorf("wago_gpu_v1 scalar access: status %d", status)})
}
func (p *Plugin) bufferCall(caller wago.Caller, call wago.HostCall, name string) {
	scalar := name == "getBuffer" || strings.HasPrefix(name, "readBuffer") || strings.HasPrefix(name, "writeBuffer")
	var started, queued time.Time
	if !scalar || name != "getBuffer" {
		started = time.Now()
	}
	id, err := p.resolver.Resolve(caller)
	if err != nil {
		bufferTrap(V1InvalidState)
	}
	mayCancel, err := p.resolver.InvocationMayCancel(caller)
	if err != nil {
		bufferTrap(V1InvalidState)
	}
	var ctx context.Context = context.Background()
	if mayCancel {
		ctx, err = p.resolver.InvocationContext(caller)
		if err != nil {
			bufferTrap(V1InvalidState)
		}
	}
	if !scalar {
		queued = time.Now()
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.observeBufferDeviceLoss()
	var queueWait time.Duration
	if !scalar {
		queueWait = time.Since(queued)
	}
	instance := p.buffers.instances[id]
	// Ordinary CPU scalar loops need no timer or per-element heap allocation.
	needsDeadline := name == "dispatch" || strings.HasPrefix(name, "setBuffer") || strings.HasPrefix(name, "copyBuffer")
	if !needsDeadline && instance != nil && (strings.HasPrefix(name, "readBuffer") || strings.HasPrefix(name, "writeBuffer")) {
		b := instance.buffers[uint32(call.I32(0))]
		needsDeadline = b != nil && !b.cpuCurrent
	}
	if needsDeadline {
		if started.IsZero() {
			started = time.Now()
		}
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(ctx, started.Add(p.config.RunTimeout))
		defer cancel()
	}
	if scalar {
		if p.stopped || instance == nil {
			bufferTrap(V1InvalidState)
		}
		if name == "getBuffer" {
			slot := uint32(call.I32(0))
			if slot >= 8 || instance.bindings[slot] == 0 {
				bufferTrap(V1InvalidBinding)
			}
			if ctx.Err() != nil {
				bufferTrap(V1Cancelled)
			}
			call.SetI32(0, int32(instance.bindings[slot]))
			return
		}
		p.scalarBuffer(ctx, instance, call, name)
		return
	}
	op := BufferOperation{Instance: id, Import: name, QueueWait: queueWait, TransferCountsComplete: true, ProfileValid: p.config.ProfileStages, ReasonCode: "NONE", Outcome: "OK"}
	status := V1OK
	defer func() {
		if !p.config.ProfileStages || (status != V1OK && status != V1CPUFallback) || !op.TransferCountsComplete {
			op.ProfileValid = false
			op.Upload = 0
			op.DeviceCopy = 0
			op.Compute = 0
			op.Download = 0
			op.Conversion = 0
			op.Commit = 0
		}
		switch status {
		case V1LimitExceeded:
			op.ReasonCode = "RESOURCE_LIMIT"
		case V1Cancelled:
			op.ReasonCode = "CANCELLED"
		case V1ContentsLost:
			op.ReasonCode = "CONTENTS_LOST"
		case V1DeviceError:
			op.ReasonCode = "DEVICE_FAILURE"
		}
		op.Status = status
		op.Total = time.Since(started)
		if status != V1OK && status != V1CPUFallback {
			op.Outcome = "ERROR"
			if op.ReasonCode == "NONE" {
				op.ReasonCode = "CONTRACT_ERROR"
			}
		}
		if p.buffers.stats.Last.Sequence == math.MaxUint64 {
			panic(wago.HostTrap{Err: fmt.Errorf("diagnostic sequence exhausted")})
		}
		op.Sequence = p.buffers.stats.Last.Sequence + 1
		p.buffers.stats.Last = op
	}()
	if p.stopped || instance == nil {
		switch name {
		case "createBuffer":
			call.SetI32(0, 0)
			call.SetI32(1, V1InvalidState)
		case "createBufferPacked":
			call.SetI64(0, int64(uint64(uint32(V1InvalidState))<<32))
		default:
			call.SetI32(0, V1InvalidState)
		}
		status = V1InvalidState
		return
	}
	switch name {
	case "createBuffer", "createBufferPacked":
		typ, count := ElementType(uint32(call.I32(0))), uint32(call.I32(1))
		op.Count = count
		handle := uint32(0)
		size := typ.spec().size
		switch {
		case size == 0:
			status = V1TypeMismatch
		case count > p.config.MaxElements || uint32(len(instance.buffers)) >= p.config.MaxBuffersPerInstance || p.buffers.nextHandle >= math.MaxUint32:
			status = V1LimitExceeded
		case ctx.Err() != nil:
			status = V1Cancelled
		case !p.reserveBuffer(instance, uint64(count)*size):
			status = V1LimitExceeded
		default:
			p.buffers.nextHandle++
			handle = uint32(p.buffers.nextHandle)
			instance.buffers[handle] = &bufferState{typ: typ, count: count, cpu: make([]byte, uint64(count)*size), version: 1, cpuCurrent: true}
		}
		if name == "createBufferPacked" {
			call.SetI64(0, int64(uint64(uint32(status))<<32|uint64(handle)))
		} else {
			call.SetI32(0, int32(handle))
			call.SetI32(1, status)
		}
	case "freeBuffer":
		handle := uint32(call.I32(0))
		b := instance.buffers[handle]
		if b == nil {
			status = V1InvalidHandle
		} else if ctx.Err() != nil {
			status = V1Cancelled
		} else {
			p.closeBufferResource(instance, b.gpu, physicalBytes(b))
			delete(instance.buffers, handle)
			for i, h := range instance.bindings {
				if h == handle {
					instance.bindings[i] = 0
				}
			}
			p.releaseBufferBytes(instance, uint64(len(b.cpu)))
		}
		call.SetI32(0, status)
	case "bindBuffer":
		slot, handle := uint32(call.I32(0)), uint32(call.I32(1))
		if slot >= 8 {
			status = V1InvalidBinding
		} else if handle != 0 && instance.buffers[handle] == nil {
			status = V1InvalidHandle
		} else if ctx.Err() != nil {
			status = V1Cancelled
		} else {
			instance.bindings[slot] = handle
		}
		call.SetI32(0, status)
	case "dispatch":
		kernel, count := uint32(call.I32(0)), uint32(call.I32(1))
		op.KernelID = kernel
		op.Count = count
		status = p.dispatchBuffers(ctx, instance, kernel, count, &op)
		call.SetI32(0, status)
	default:
		status = p.transferBuffer(ctx, caller, instance, call, name, &op)
		call.SetI32(0, status)
	}
}
func (p *Plugin) scalarBuffer(ctx context.Context, i *bufferInstance, call wago.HostCall, name string) {
	handle, index := uint32(call.I32(0)), uint32(call.I32(1))
	b := i.buffers[handle]
	if b == nil {
		bufferTrap(V1InvalidHandle)
	}
	write := strings.HasPrefix(name, "write")
	prefix := "readBuffer"
	if write {
		prefix = "writeBuffer"
	}
	if strings.TrimPrefix(name, prefix) != b.typ.spec().suffix {
		bufferTrap(V1TypeMismatch)
	}
	if index >= b.count {
		bufferTrap(V1InvalidRange)
	}
	if ctx.Err() != nil {
		bufferTrap(V1Cancelled)
	}
	if status := p.ensureCPU(ctx, i, b, nil); status != V1OK {
		bufferTrap(status)
	}
	if ctx.Err() != nil {
		bufferTrap(V1Cancelled)
	}
	size := b.typ.spec().size
	data := b.cpu[uint64(index)*size : uint64(index+1)*size]
	if write {
		switch b.typ {
		case TypeI8, TypeU8:
			data[0] = byte(call.I32(2))
		case TypeI16, TypeU16:
			binary.LittleEndian.PutUint16(data, uint16(call.I32(2)))
		case TypeI32, TypeU32:
			binary.LittleEndian.PutUint32(data, uint32(call.I32(2)))
		case TypeF16:
			binary.LittleEndian.PutUint16(data, floatToHalf(math.Float32bits(call.F32(2))))
		case TypeF32:
			binary.LittleEndian.PutUint32(data, math.Float32bits(call.F32(2)))
		case TypeI64, TypeU64:
			binary.LittleEndian.PutUint64(data, uint64(call.I64(2)))
		case TypeF64:
			binary.LittleEndian.PutUint64(data, math.Float64bits(call.F64(2)))
		}
		b.version++
		b.gpuCurrent = false
		return
	}
	switch b.typ {
	case TypeI8:
		call.SetI32(0, int32(int8(data[0])))
	case TypeU8:
		call.SetI32(0, int32(data[0]))
	case TypeI16:
		call.SetI32(0, int32(int16(binary.LittleEndian.Uint16(data))))
	case TypeU16:
		call.SetI32(0, int32(binary.LittleEndian.Uint16(data)))
	case TypeI32, TypeU32:
		call.SetI32(0, int32(binary.LittleEndian.Uint32(data)))
	case TypeF16:
		call.SetF32(0, math.Float32frombits(halfToFloat(binary.LittleEndian.Uint16(data))))
	case TypeF32:
		call.SetF32(0, math.Float32frombits(binary.LittleEndian.Uint32(data)))
	case TypeI64, TypeU64:
		call.SetI64(0, int64(binary.LittleEndian.Uint64(data)))
	case TypeF64:
		call.SetF64(0, math.Float64frombits(binary.LittleEndian.Uint64(data)))
	}
}
func (p *Plugin) dispatchBuffers(ctx context.Context, i *bufferInstance, id, count uint32, op *BufferOperation) int32 {
	if i.module == nil || !i.module.verified {
		op.ReasonCode = "CONTRACT_UNVERIFIED"
		return V1InvalidKernel
	}
	k := i.module.kernels[id]
	if k == nil || k.invalid != nil {
		op.ReasonCode = "CONTRACT_ERROR"
		return V1InvalidKernel
	}
	var seen [8]uint32
	seenCount := 0
	for _, slot := range k.config.Bindings {
		handle := i.bindings[slot.Slot]
		b := i.buffers[handle]
		if b == nil {
			return V1InvalidBinding
		}
		for _, previous := range seen[:seenCount] {
			if previous == handle {
				return V1InvalidBinding
			}
		}
		seen[seenCount] = handle
		seenCount++
		if b.typ != slot.Type {
			return V1TypeMismatch
		}
		if count > b.count {
			return V1InvalidRange
		}
	}
	if count > p.config.MaxElements {
		return V1LimitExceeded
	}
	if count == 0 {
		op.Outcome = "NOOP"
		return V1OK
	}
	if ctx.Err() != nil {
		op.ReasonCode = "CANCELLED"
		return V1Cancelled
	}
	op.Outcome = "CPU_FALLBACK_READY"
	op.ReasonCode = "NO_DEVICE"
	if p.config.Disabled {
		op.ReasonCode = "DISABLED"
	} else if k.rejection != nil {
		op.ReasonCode = "UNSUPPORTED_KERNEL"
		op.Reason = k.rejection.Error()
	} else if k.lowered.float && !k.config.RelaxedFloat {
		op.ReasonCode = "FLOAT_MODE"
	} else if count < k.config.MinElements {
		op.ReasonCode = "BELOW_THRESHOLD"
	}
	if !p.config.Disabled && k.rejection == nil && (!k.lowered.float || k.config.RelaxedFloat) && count >= k.config.MinElements && k.pipeline != nil && !p.stats.GPUFailed {
		status := p.executeBufferKernel(ctx, i, k, count, op)
		if status != V1DeviceError && status != V1ContentsLost {
			return status
		}
		for _, slot := range k.config.Bindings {
			if ready := p.ensureCPU(ctx, i, i.buffers[i.bindings[slot.Slot]], op); ready != V1OK {
				return ready
			}
		}
		if ctx.Err() != nil {
			return V1Cancelled
		}
		op.Outcome = "CPU_FALLBACK_READY"
		return V1CPUFallback
	}
	if k.pipelineError != nil {
		op.ReasonCode = "PIPELINE_FAILURE"
		op.Reason = k.pipelineError.Error()
	}
	for _, slot := range k.config.Bindings {
		if status := p.ensureCPU(ctx, i, i.buffers[i.bindings[slot.Slot]], op); status != V1OK {
			return status
		}
	}
	if ctx.Err() != nil {
		return V1Cancelled
	}
	return V1CPUFallback
}
