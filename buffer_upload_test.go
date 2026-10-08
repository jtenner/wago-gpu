package wagogpu

import (
	"context"
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

type rangedFakeDevice struct {
	*fakeBufferDevice
	batches      int
	retained     uint64
	afterSuccess func()
}

func (d *rangedFakeDevice) RetainedBytes() uint64 { return d.retained }
func (d *rangedFakeDevice) UploadBufferRange(ctx context.Context, b deviceBuffer, offset uint64, data []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	copy(b.(*fakeDeviceBuffer).bytes[offset:offset+uint64(len(data))], data)
	return nil
}
func (d *rangedFakeDevice) ExecuteBuffersWithUploads(ctx context.Context, pipeline bufferPipeline, uniform deviceBuffer, resources []deviceBuffer, seeds []bufferSeed, uploads []bufferUpload, count uint32, parameterUpload bool, op *BufferOperation) error {
	d.batches++
	for _, upload := range uploads {
		if err := d.UploadBufferRange(ctx, upload.resource, upload.offset, upload.data); err != nil {
			return err
		}
	}
	err := d.ExecuteBuffers(ctx, pipeline, uniform, resources, seeds, count, parameterUpload, op)
	if err != nil {
		for _, upload := range uploads {
			d.retained += uint64(len(upload.data))
		}
		if parameterUpload {
			d.retained += 16
		}
	} else if d.afterSuccess != nil {
		d.afterSuccess()
	}
	return err
}

func TestBufferDirtyRanges(t *testing.T) { testBufferDirtyRanges(t, false) }
func testBufferDirtyRanges(t *testing.T, hardware bool) {
	for _, profile := range []bool{false, true} {
		t.Run(map[bool]string{false: "combined", true: "profiled"}[profile], func(t *testing.T) {
			cfg := bufferConfig()
			cfg.Disabled, cfg.ProfileStages = false, profile
			cfg.Kernels = cfg.Kernels[:1]
			d := &rangedFakeDevice{fakeBufferDevice: &fakeBufferDevice{}}
			var factory func() (backend, error)
			if !hardware {
				factory = func() (backend, error) { return d, nil }
			}
			rt, p := setup(t, cfg, factory)
			source := testModule(`(call $write (call $get (i32.const 1)) (local.get $i)
 (f32.mul (call $read (call $get (i32.const 0)) (local.get $i)) (f32.const 2)))`)
			source = strings.TrimSuffix(source, ")") + `(export "writeValue" (func $write)))`
			_, in := instance(t, rt, wat(t, source))
			const count = 257
			for slot := uint64(0); slot < 2; slot++ {
				r, err := in.Invoke("create", uint64(TypeF32), count)
				if err != nil || r[0]>>32 != 0 {
					t.Fatal(r, err)
				}
				invoke(t, in, "bind", slot, uint64(uint32(r[0])))
			}
			for j := uint32(0); j < count; j++ {
				in.WriteFloat32Le(j*4, float32(j+1))
			}
			invoke(t, in, "set", 1, 0, 0, 0, count)
			if invoke(t, in, "dispatch", 1, count) != V1OK {
				t.Fatal(p.BufferSnapshot())
			}
			if p.BufferSnapshot().Last.GPUUploadBytes != count*4 {
				t.Fatal("unknown base did not upload in full")
			}
			in.WriteFloat32Le(0, 37)
			invoke(t, in, "set", 1, 32, 0, 0, 1)
			in.WriteFloat32Le(0, 83)
			invoke(t, in, "set", 1, 64, 0, 0, 1)
			_, err := in.Invoke("writeValue", 1, 95, uint64(math.Float32bits(111)))
			if err != nil {
				t.Fatal(err)
			}
			if invoke(t, in, "set", 1, count-1, 0, 0, 2) != V1InvalidRange {
				t.Fatal("invalid range accepted")
			}
			if invoke(t, in, "dispatch", 1, count) != V1OK {
				t.Fatal(p.BufferSnapshot())
			}
			op := p.BufferSnapshot().Last
			if op.UploadCount != 1 || op.GPUUploadBytes != 64*4 || op.LogicalUploadBytes != 64*4 {
				t.Fatal("dirty union traffic", op)
			}
			if invoke(t, in, "copy", 2, 0, 0, 0, count) != V1OK {
				t.Fatal("copy failed")
			}
			for j := uint32(0); j < count; j++ {
				want := float32(j+1) * 2
				switch j {
				case 32:
					want = 74
				case 64:
					want = 166
				case 95:
					want = 222
				}
				got, _ := in.ReadFloat32Le(j * 4)
				if got != want {
					t.Fatal(j, got, want)
				}
			}
			if invoke(t, in, "dispatch", 1, count) != V1OK || p.BufferSnapshot().Last.UploadCount != 0 {
				t.Fatal("unchanged input uploaded again")
			}
			if !hardware && (d.batches > 0) == profile {
				t.Fatal("wrong upload mode", d.batches, profile)
			}
		})
	}
}

func TestNarrowDirtyUpload(t *testing.T) {
	d := &rangedFakeDevice{fakeBufferDevice: &fakeBufferDevice{}}
	cfg := bufferConfig()
	cfg.Disabled = false
	rt, p := setup(t, cfg, func() (backend, error) { return d, nil })
	_, in := instance(t, rt, wat(t, testModule(``)))
	for _, typ := range []ElementType{TypeI8, TypeU16, TypeF16} {
		r, err := in.Invoke("create", uint64(typ), 65)
		if err != nil {
			t.Fatal(err)
		}
		func() {
			p.mu.Lock()
			defer p.mu.Unlock()
			for _, i := range p.buffers.instances {
				b := i.buffers[uint32(r[0])]
				for j := range b.cpu {
					b.cpu[j] = byte(j + 1)
				}
				if p.ensureGPU(context.Background(), i, b, &BufferOperation{}) != V1OK {
					t.Fatal("full upload failed")
				}
				old := append([]byte(nil), b.gpu.(*fakeDeviceBuffer).bytes...)
				stride := b.typ.spec().size
				b.cpu[31*stride] = 0x81
				b.dirty(31, 1)
				op := BufferOperation{}
				if p.ensureGPU(context.Background(), i, b, &op) != V1OK {
					t.Fatal("patch failed")
				}
				if op.GPUUploadBytes != 4 || op.LogicalUploadBytes != stride {
					t.Fatal(op)
				}
				data := b.gpu.(*fakeDeviceBuffer).bytes
				for j := 0; j < 65; j++ {
					want := binary.LittleEndian.Uint32(old[j*4:])
					if j == 31 {
						want = uint32(b.cpu[31*stride])
						if stride == 2 {
							want = uint32(binary.LittleEndian.Uint16(b.cpu[31*stride:]))
						}
					}
					if got := binary.LittleEndian.Uint32(data[j*4:]); got != want {
						t.Fatal(typ, j, got, want)
					}
				}
			}
		}()
	}
}

func TestBatchedUploadFailureAndCancellation(t *testing.T) {
	for _, mode := range []string{"preflight", "device-error", "cancel-after-completion"} {
		t.Run(mode, func(t *testing.T) {
			d := &rangedFakeDevice{fakeBufferDevice: &fakeBufferDevice{}}
			cfg := bufferConfig()
			cfg.Disabled = false
			cfg.Kernels = cfg.Kernels[:1]
			rt, p := setup(t, cfg, func() (backend, error) { return d, nil })
			_, in := instance(t, rt, wat(t, testModule(`(call $write (call $get (i32.const 1)) (local.get $i) (f32.mul (call $read (call $get (i32.const 0)) (local.get $i)) (f32.const 2)))`)))
			for slot := uint64(0); slot < 2; slot++ {
				r, _ := in.Invoke("create", uint64(TypeF32), 4)
				invoke(t, in, "bind", slot, uint64(uint32(r[0])))
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			func() {
				p.mu.Lock()
				defer p.mu.Unlock()
				for _, i := range p.buffers.instances {
					k := i.module.kernels[1]
					want := V1Cancelled
					switch mode {
					case "preflight":
						p.config.MaxInstanceBufferBytes = 64
						want = V1LimitExceeded
					case "device-error":
						d.errorAfterCompletion = true
						want = V1DeviceError
					case "cancel-after-completion":
						d.afterSuccess = cancel
					}
					if got := p.executeBufferKernel(ctx, i, k, 4, &BufferOperation{}); got != want {
						t.Fatal(got, want)
					}
					if i.buffers[2].version != 1 || !i.buffers[2].cpuCurrent {
						t.Fatal("failed output published")
					}
					if mode == "preflight" && (d.batches != 0 || i.bytes != 48) {
						t.Fatal("preflight queued or leaked payload", d.batches, i.bytes)
					}
					if mode == "device-error" && (d.retained != 32 || i.retiredBytes < 32) {
						t.Fatal("uncertain payload charge lost", d.retained, i.retiredBytes)
					}
					if mode == "cancel-after-completion" && (!i.buffers[1].gpuCurrent || i.retiredBytes != 0) {
						t.Fatal("completed input cache or payload wrong")
					}
				}
			}()
			if err := rt.CloseContext(context.Background()); err != nil {
				t.Fatal(err)
			}
			if p.BufferSnapshot().RuntimeBufferBytes != 0 || d.buffers != 0 {
				t.Fatal("cleanup leaked", p.BufferSnapshot(), d.buffers, d.closes)
			}
		})
	}
}
