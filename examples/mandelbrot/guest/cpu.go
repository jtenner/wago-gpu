//go:build tinygo && !gpu

package main

// This is a complete WASI program with no GPU imports.
func render(width, height, iterations uint32) []byte {
	pixels := make([]byte, width*height)
	for row := uint32(0); row < height; row++ {
		for col := uint32(0); col < width; col++ {
			cr, ci := point(col, row, width, height)
			var x, y float32
			for pass := uint32(1); pass <= iterations; pass++ {
				nextX, nextY := x*x-y*y+cr, 2*x*y+ci
				x, y = nextX, nextY
				if x*x+y*y > 4 {
					pixels[row*width+col] = shade(pass, iterations)
					break
				}
			}
		}
	}
	text(2, "Mandelbrot: direct Wago CPU\n")
	return pixels
}
