package wagogpu

import (
	"bytes"
	"strings"
	"testing"

	"github.com/jtenner/wago-gpu/internal/fixtures"
)

func TestTranslation(t *testing.T) {
	for _, tc := range []struct {
		name string
		wasm []byte
		want []string
	}{
		{"twice", fixtures.Twice, []string{"bitcast<f32>(0x40000000u)", "t0 * t1", "return t2;"}},
		{"square", fixtures.Square, []string{"t0 * t1", "bitcast<f32>(0x3f800000u)", "t2 + t3", "return t4;"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, e := CompileWGSL(tc.wasm, "kernel")
			if e != nil {
				t.Fatal(e)
			}
			for _, w := range tc.want {
				if !strings.Contains(s, w) {
					t.Fatalf("missing %q in %s", w, s)
				}
			}
		})
	}
	// Operand order must survive stack translation, including nested subtraction.
	s, e := compileBody([]byte{0, 0x20, 0, 0x43, 0, 0, 0, 0x40, 0x93, 0x43, 0, 0, 0x80, 0x3f, 0x94, 0x0b})
	if e != nil || !strings.Contains(s, "t0 - t1") || !strings.Contains(s, "t2 * t3") {
		t.Fatalf("%s: %v", s, e)
	}
}
func TestUnsupportedAndMalformed(t *testing.T) {
	for _, b := range [][]byte{
		{0, 0x20, 0, 0x20, 0, 0x95, 0x0b},               // division
		{0, 0x20, 1, 0x0b}, {1, 1, 0x7d, 0x20, 0, 0x0b}, // other locals
		{0, 0x92, 0x0b}, {0, 0x20, 0}, {0, 0x0b}, {0, 0x20, 0, 0x20, 0, 0x0b},
		{0, 0x20, 0, 0x0b, 0}, {0, 0x43, 1}, {0, 0x20, 0x80},
		{0, 0x10, 0, 0x0b}, // call
	} {
		if _, e := compileBody(b); e == nil {
			t.Errorf("accepted %x", b)
		}
	}
	for _, b := range [][]byte{nil, {0, 97, 115, 109, 1, 0, 0, 0, 1, 0xff, 0xff, 0xff, 0xff, 0x1f}, fixtures.Twice[:len(fixtures.Twice)-1], bytes.Repeat([]byte{0}, maxModuleBytes+1)} {
		if _, e := CompileWGSL(b, "kernel"); e == nil {
			t.Fatal("accepted malformed module")
		}
	}
	if _, e := CompileWGSL(fixtures.Twice, "cpu"); e == nil {
		t.Fatal("accepted wrong signature")
	}
	if _, e := CompileWGSL(fixtures.Twice, "missing"); e == nil {
		t.Fatal("accepted missing export")
	}
	long := append([]byte{0}, bytes.Repeat([]byte{0x20, 0}, maxInstructions+1)...)
	long = append(long, 0x0b)
	if _, e := compileBody(long); e == nil {
		t.Fatal("accepted large kernel")
	}
}
func FuzzCompileWGSL(f *testing.F) {
	f.Add(fixtures.Twice)
	f.Add(fixtures.Square)
	f.Add(fixtures.Heavy)
	f.Add([]byte{0, 97, 115, 109, 1, 0, 0, 0})
	f.Fuzz(func(t *testing.T, b []byte) { CompileWGSL(b, "kernel") })
}

func TestHeavyTranslation(t *testing.T) {
	s, e := CompileWGSL(fixtures.Heavy, "kernel")
	if e != nil {
		t.Fatal(e)
	}
	if strings.Count(s, " * ") != 32 || strings.Count(s, " + ") != 32 {
		t.Fatal("heavy Wasm instructions were not translated")
	}
}
