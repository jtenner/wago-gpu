package wagogpu_test

import (
	"errors"
	"fmt"
	gpu "github.com/jtenner/wago-gpu"
	"os"
	"testing"
)

func TestPublicCompileError(t *testing.T) {
	source, e := os.ReadFile("internal/fixtures/buffers.wasm")
	if e != nil {
		t.Fatal(e)
	}
	base := gpu.KernelConfig{ID: 1, Export: "wago_gpu.kernel.double", CPUExport: "wago_gpu.cpu.double", Bindings: []gpu.BindingConfig{{Slot: 0, Type: gpu.TypeF32, Access: gpu.AccessRead}, {Slot: 1, Type: gpu.TypeF32, Access: gpu.AccessWrite}}}
	wrong := base
	wrong.Export = "wago_gpu.kernel.missing"
	wrong.CPUExport = "wago_gpu.cpu.missing"
	unsupported := append([]byte(nil), source...)
	// Change the selected kernel's multiply to divide, preserving valid Wasm.
	changed := false
	for i := 0; i+2 < len(unsupported); i++ {
		if unsupported[i] == 0x94 && unsupported[i+1] == 0x10 {
			unsupported[i] = 0x95
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("fixture has no multiply followed by store call")
	}
	for _, test := range []struct {
		name   string
		source []byte
		k      gpu.KernelConfig
		kind   gpu.CompileErrorKind
	}{{"signature", source, wrong, gpu.CompileInvalidContract}, {"instruction", unsupported, base, gpu.CompileUnsupported}, {"limit", make([]byte, (4<<20)+1), base, gpu.CompileLimit}} {
		t.Run(test.name, func(t *testing.T) {
			_, err := gpu.CompileBufferWGSL(test.source, test.k)
			for _, e := range []error{err, fmt.Errorf("wrapped: %w", err)} {
				var c *gpu.CompileError
				if !errors.As(e, &c) || c.Kind != test.kind {
					t.Fatalf("category: %v", e)
				}
			}
		})
	}
}

func TestPublicLocalVectorLimit(t *testing.T) {
	leb := func(n int) []byte {
		var b []byte
		for n >= 128 {
			b = append(b, byte(n)|128)
			n >>= 7
		}
		return append(b, byte(n))
	}
	section := func(id byte, p []byte) []byte { return append(append([]byte{id}, leb(len(p))...), p...) }
	name := func(s string) []byte { return append(leb(len(s)), []byte(s)...) }
	source := []byte{0, 97, 115, 109, 1, 0, 0, 0}
	source = append(source, section(1, []byte{1, 0x60, 1, 0x7f, 0})...)
	source = append(source, section(3, []byte{2, 0, 0})...)
	exports := []byte{2}
	exports = append(exports, name("wago_gpu.kernel.double")...)
	exports = append(exports, 0, 0)
	exports = append(exports, name("wago_gpu.cpu.double")...)
	exports = append(exports, 0, 1)
	source = append(source, section(7, exports)...)
	// An excessive group count must fail before trying to expand or read it.
	body := append(leb(16385), 0x0b)
	code := append([]byte{2}, leb(len(body))...)
	code = append(code, body...)
	code = append(code, 2, 0, 0x0b)
	source = append(source, section(10, code)...)
	_, err := gpu.CompileBufferWGSL(source, gpu.KernelConfig{ID: 1, Export: "wago_gpu.kernel.double", CPUExport: "wago_gpu.cpu.double", Bindings: []gpu.BindingConfig{{Slot: 0, Type: gpu.TypeF32, Access: gpu.AccessRead}}})
	var typed *gpu.CompileError
	if !errors.As(fmt.Errorf("outer: %w", err), &typed) || typed.Kind != gpu.CompileLimit {
		t.Fatalf("category: %v", err)
	}
}
