// Package settings contains the limits shared by the host and Wasm guests.
package settings

const (
	DefaultWidth      = 1920
	DefaultHeight     = 1080
	DefaultIterations = 64
	MaxDimension      = 4096
	MaxPixels         = 4_194_304
	MaxIterations     = 128
)
