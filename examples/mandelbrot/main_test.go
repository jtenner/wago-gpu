package main

import (
	"bytes"
	"errors"
	"fmt"
	gpu "github.com/jtenner/wago-gpu"
	"github.com/jtenner/wago-gpu/examples/internal/runwasm"
	"github.com/jtenner/wago-gpu/examples/mandelbrot/internal/settings"
	"github.com/wago-org/wasi/p1"
	"io"
	"testing"
)

func image(t *testing.T, program string, cpu, required bool, width, height, iterations uint32) []byte {
	t.Helper()
	var out, stderr bytes.Buffer
	if e := run(program, cpu, required, width, height, iterations, &out, &stderr); e != nil {
		t.Fatal(e, stderr.String())
	}
	header := []byte(fmt.Sprintf("P5\n%d %d\n255\n", width, height))
	if !bytes.HasPrefix(out.Bytes(), header) || out.Len() != len(header)+int(width*height) {
		t.Fatalf("invalid PGM output: %d bytes", out.Len())
	}
	return bytes.Clone(out.Bytes()[len(header):])
}
func TestKnownMandelbrotPoints(t *testing.T) {
	pixels := image(t, "cpu", true, false, 6, 5, 16)
	// The middle row is real c = -1.75,-1.25,-0.75,-0.25,0.25,0.75.
	// Its first five points remain bounded. c=0.75 first escapes on pass 3.
	for i := 0; i < 5; i++ {
		if pixels[12+i] != 0 {
			t.Fatal("real-axis bounded point escaped", i, pixels[12+i])
		}
	}
	if pixels[17] != byte(255-3*255/16) {
		t.Fatal("wrong escape count for c=0.75", pixels[17])
	}
	for row := 0; row < 2; row++ {
		if !bytes.Equal(pixels[row*6:(row+1)*6], pixels[(4-row)*6:(5-row)*6]) {
			t.Fatal("conjugate points have different colors")
		}
	}
}
func TestBufferCPUImage(t *testing.T) {
	for _, size := range [][3]uint32{{1, 1, 1}, {17, 9, 8}, {19, 17, 12}, {32, 24, 16}} {
		t.Run(fmt.Sprint(size), func(t *testing.T) {
			want := image(t, "cpu", true, false, size[0], size[1], size[2])
			got := image(t, "buffers", true, false, size[0], size[1], size[2])
			if !bytes.Equal(got, want) {
				t.Fatal("buffer CPU differs from direct CPU")
			}
		})
	}
}
func TestMandelbrotKernel(t *testing.T) {
	source, e := programs.ReadFile("gpu.wasm")
	if e != nil {
		t.Fatal(e)
	}
	shader, e := gpu.CompileBufferWGSL(source, kernel())
	if e != nil {
		t.Fatal(e)
	}
	for _, store := range []string{"slot2[index] =", "slot3[index] ="} {
		if !bytes.Contains([]byte(shader), []byte(store)) {
			t.Fatal("missing actual Wasm output", store)
		}
	}
}
func TestInvalidSettings(t *testing.T) {
	for _, test := range []struct {
		program       string
		cpu, required bool
		w, h, n       uint32
	}{
		{"other", false, false, 1, 1, 1}, {"cpu", false, false, 0, 1, 1},
		{"cpu", false, false, 4097, 1, 1}, {"cpu", false, false, 1, 4097, 1},
		{"cpu", false, false, 4096, 4096, 1}, {"cpu", false, false, 1, 1, 129}, {"cpu", false, false, 1, 1, 0},
		{"cpu", false, true, 1, 1, 1}, {"buffers", true, true, 1, 1, 1},
	} {
		if e := run(test.program, test.cpu, test.required, test.w, test.h, test.n, io.Discard, io.Discard); e == nil {
			t.Fatal("invalid settings accepted", test)
		}
	}
}
func TestGuestArgumentLimits(t *testing.T) {
	source, e := programs.ReadFile("cpu.wasm")
	if e != nil {
		t.Fatal(e)
	}
	for _, args := range [][]string{{"mandelbrot", "0", "8", "16"}, {"mandelbrot", "4097", "8", "16"}, {"mandelbrot", "8", "8", "129"}, {"mandelbrot", "bad", "8", "16"}, {"mandelbrot", "4096", "4096", "1"}} {
		var out bytes.Buffer
		e := runwasm.Run(source, runwasm.Options{WASI: &p1.Config{Args: args, Stdout: &out, Stderr: io.Discard}, Calls: []string{"_start"}}, nil)
		if e == nil || out.Len() != 0 {
			t.Fatal("invalid standalone guest arguments accepted", args, e)
		}
	}
}

type failedOutput struct{}

func (failedOutput) Write([]byte) (int, error) { return 0, errors.New("injected output failure") }
func TestWASIOutputFailure(t *testing.T) {
	if e := run("cpu", true, false, 6, 5, 16, failedOutput{}, io.Discard); e == nil {
		t.Fatal("WASI output error was lost")
	}
}

func TestLargeCPUImages(t *testing.T) {
	for _, size := range [][2]uint32{{settings.DefaultWidth, settings.DefaultHeight}, {2560, 1440}, {settings.MaxDimension, 1}} {
		image(t, "cpu", false, false, size[0], size[1], 1)
	}
}
func TestStandaloneDefaultImage(t *testing.T) {
	source, e := programs.ReadFile("cpu.wasm")
	if e != nil {
		t.Fatal(e)
	}
	var out bytes.Buffer
	e = runwasm.Run(source, runwasm.Options{WASI: &p1.Config{Args: []string{"mandelbrot"}, Stdout: &out, Stderr: io.Discard}, Calls: []string{"_start"}}, nil)
	if e != nil {
		t.Fatal(e)
	}
	header := []byte(fmt.Sprintf("P5\n%d %d\n255\n", settings.DefaultWidth, settings.DefaultHeight))
	if !bytes.HasPrefix(out.Bytes(), header) || out.Len() != len(header)+settings.DefaultWidth*settings.DefaultHeight {
		t.Fatal("invalid default full-HD image")
	}
}
