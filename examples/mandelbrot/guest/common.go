//go:build tinygo

package main

import "github.com/jtenner/wago-gpu/examples/mandelbrot/internal/settings"

func number(text string, max uint32) uint32 {
	var n uint32
	for _, c := range text {
		if c < '0' || c > '9' {
			panic("arguments must be positive integers")
		}
		n = n*10 + uint32(c-'0')
		if n > max {
			panic("argument exceeds its limit")
		}
	}
	if n == 0 {
		panic("arguments must be positive integers")
	}
	return n
}

func point(x, y, width, height uint32) (float32, float32) {
	return -2 + 3*(float32(x)+0.5)/float32(width), -1.25 + 2.5*(float32(y)+0.5)/float32(height)
}

func shade(escaped, iterations uint32) byte {
	if escaped == 0 {
		return 0
	}
	return byte(255 - escaped*255/iterations)
}

func main() {
	width, height, iterations := uint32(settings.DefaultWidth), uint32(settings.DefaultHeight), uint32(settings.DefaultIterations)
	args := arguments()
	if len(args) != 1 {
		if len(args) != 4 {
			panic("usage: mandelbrot [width height iterations]")
		}
		width, height, iterations = number(args[1], settings.MaxDimension), number(args[2], settings.MaxDimension), number(args[3], settings.MaxIterations)
	}
	if uint64(width)*uint64(height) > settings.MaxPixels {
		panic("image exceeds the pixel limit")
	}
	pixels := render(width, height, iterations)
	// WASI fd_write carries the PGM header and pixel bytes to stdout.
	text(1, "P5\n")
	digits(1, width)
	text(1, " ")
	digits(1, height)
	text(1, "\n255\n")
	write(1, pixels)
}
