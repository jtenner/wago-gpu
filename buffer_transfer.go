package wagogpu

import (
	"context"
	wago "github.com/wago-org/wago"
	"strings"
	"time"
)

func gcStorage(t ElementType) wago.GuestGCArrayStorage {
	switch t {
	case TypeI8, TypeU8:
		return wago.GuestGCArrayI8
	case TypeI16, TypeU16, TypeF16:
		return wago.GuestGCArrayI16
	case TypeI32, TypeU32:
		return wago.GuestGCArrayI32
	case TypeI64, TypeU64:
		return wago.GuestGCArrayI64
	case TypeF32:
		return wago.GuestGCArrayF32
	case TypeF64:
		return wago.GuestGCArrayF64
	}
	return 0
}
func importCancelled(ctx context.Context, deadline time.Time) bool {
	return ctx.Err() != nil || (!deadline.IsZero() && !time.Now().Before(deadline))
}

func (p *Plugin) transferBuffer(ctx context.Context, deadline time.Time, caller wago.Caller, i *bufferInstance, call wago.HostCall, name string, op *BufferOperation) int32 {
	handle, offset, count := uint32(call.I32(0)), uint32(call.I32(1)), uint32(call.I32(4))
	op.Count = count
	b := i.buffers[handle]
	if b == nil {
		return V1InvalidHandle
	}
	if uint64(offset)+uint64(count) > uint64(b.count) {
		return V1InvalidRange
	}
	size := b.typ.spec().size
	storage, ok := any(caller).(wago.GuestStorageHostModule)
	if !ok {
		return V1UnsupportedStorage
	}
	set := strings.HasPrefix(name, "set")
	gc := strings.HasSuffix(name, "GC")
	wide := strings.HasSuffix(name, "64")
	access := wago.GuestStorageWrite
	if set {
		access = wago.GuestStorageRead
	}
	// Obtain parameters before results overwrite their slots. No view or GCRef
	// survives WithGuestStorage. A synchronous readback can occur within the
	// borrow; its native callbacks capture only plugin-owned storage.
	raw, _ := call.RawParam(2)
	memoryIndex := uint32(raw)
	var byteOffset uint64
	if wide {
		byteOffset = uint64(call.I64(3))
	} else {
		byteOffset = uint64(uint32(call.I32(3)))
	}

	status := V1OK
	err := storage.WithGuestStorage(func(s wago.GuestStorage) error {
		var data []byte
		if gc {
			ref, e := s.GCRef(raw)
			if e != nil {
				status = V1UnsupportedStorage
				return nil
			}
			info, e := s.GCArrayInfo(ref)
			if e != nil {
				status = V1UnsupportedStorage
				return nil
			}
			if info.Storage != gcStorage(b.typ) {
				status = V1TypeMismatch
				return nil
			}
			if !set && !info.Mutable {
				status = V1UnsupportedStorage
				return nil
			}
			if byteOffset+uint64(count) > uint64(info.Length) {
				status = V1InvalidRange
				return nil
			}
			if count == 0 {
				return nil
			}
			if importCancelled(ctx, deadline) {
				status = V1Cancelled
				return nil
			}
			temporary := uint64(0)
			if !info.Mutable {
				temporary = uint64(info.Length) * size
				if !p.reserveBuffer(i, temporary) {
					status = V1LimitExceeded
					return nil
				}
				defer p.releaseBufferBytes(i, temporary)
			}
			data, _, e = s.GCArrayBytes(ref, access)
			if e != nil {
				status = V1UnsupportedStorage
				return nil
			}
			data = data[byteOffset*size : (byteOffset+uint64(count))*size]
		} else {
			if i.module == nil || !i.module.verified {
				status = V1UnsupportedStorage
				return nil
			}
			if uint64(memoryIndex) >= uint64(len(i.module.memories)) {
				status = V1InvalidRange
				return nil
			}
			declaration := i.module.memories[memoryIndex]
			if declaration.shared || declaration.wide != wide {
				status = V1UnsupportedStorage
				return nil
			}
			info, e := s.MemoryInfo(memoryIndex)
			if e != nil {
				status = V1UnsupportedStorage
				return nil
			}
			expected := wago.GuestMemory32
			if wide {
				expected = wago.GuestMemory64
			}
			if info.AddressType != expected {
				status = V1UnsupportedStorage
				return nil
			}
			data, e = s.MemoryRange(memoryIndex, byteOffset, uint64(count)*size, access)
			if e != nil {
				status = V1InvalidRange
				return nil
			}
			if count == 0 {
				return nil
			}
		}
		if !(set && offset == 0 && count == b.count) {
			status = p.ensureCPU(ctx, i, b, op)
			if status != V1OK {
				return nil
			}
		}
		if importCancelled(ctx, deadline) {
			status = V1Cancelled
			return nil
		}
		// Commit has no recoverable failure. Cancellation after this decision does
		// not turn a completed mutation into a failed import.
		commitStart := time.Now()
		local := b.cpu[uint64(offset)*size : (uint64(offset)+uint64(count))*size]
		if set {
			copy(local, data)
			b.version++
			b.cpuCurrent = true
			b.dirty(offset, count)
			b.lost = false
			op.GuestSetBytes = uint64(len(data))
			op.GuestSetCount = 1
			p.buffers.stats.Totals.GuestSetBytes += op.GuestSetBytes
			p.buffers.stats.Totals.GuestSetCount++
		} else {
			copy(data, local)
			op.GuestCopyBytes = uint64(len(data))
			op.GuestCopyCount = 1
			p.buffers.stats.Totals.GuestCopyBytes += op.GuestCopyBytes
			p.buffers.stats.Totals.GuestCopyCount++
		}
		op.Commit += time.Since(commitStart)
		return nil
	})
	if err != nil {
		return V1UnsupportedStorage
	}
	return status
}
