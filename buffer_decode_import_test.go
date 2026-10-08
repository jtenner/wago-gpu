package wagogpu

import (
	"errors"
	"fmt"
	"testing"
)

func TestBufferImportClassification(t *testing.T) {
	kernel := bufferConfig().Kernels[0]
	for _, tc := range []struct {
		name, module, importName, result, extra string
		want                                    CompileErrorKind
	}{
		{name: "canonical", module: "wago_gpu_v1", importName: "getBuffer", result: "i32"},
		{name: "unused_bad_signature", module: "wago_gpu_v1", importName: "getBuffer", result: "i32",
			extra: `(import "wago_gpu_v1" "readBufferF32" (func (param i32 i32) (result i64)))`},
		{name: "wrong_namespace", module: "host", importName: "getBuffer", result: "i32", want: CompileUnsupported},
		{name: "unknown_name", module: "wago_gpu_v1", importName: "unknownBuffer", result: "i32", want: CompileUnsupported},
		{name: "wrong_signature", module: "wago_gpu_v1", importName: "getBuffer", result: "f32", want: CompileInvalidContract},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source := wat(t, fmt.Sprintf(`(module
 (import %q %q (func $get (param i32) (result %s))) %s
 (func (export "wago_gpu.kernel.double") (param i32) (local $h %s) (local.set $h (call $get (i32.const 0))))
 (func (export "wago_gpu.cpu.double") (param i32)))`, tc.module, tc.importName, tc.result, tc.extra, tc.result))
			_, err := CompileBufferWGSL(source, kernel)
			if tc.want == 0 {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			var classified *CompileError
			if !errors.As(err, &classified) || classified.Kind != tc.want {
				t.Fatalf("error %v; want category %v", err, tc.want)
			}
		})
	}
}
