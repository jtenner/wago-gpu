module github.com/jtenner/wago-gpu

go 1.25

require (
	github.com/wago-org/wago v0.1.0-beta.12.0.20261007220511-7aa401f29a33
	github.com/wago-org/wasi v0.3.2-0.20261005152821-e461db531d72
)

require github.com/oliverbestmann/webgpu/libs-linux v0.0.0-20260628152803-421b8a341d08 // indirect

require (
	github.com/oliverbestmann/webgpu v1.36.0
	golang.org/x/sys v0.30.0 // indirect
)

replace github.com/oliverbestmann/webgpu => ./third_party/webgpu

replace github.com/oliverbestmann/webgpu/libs-linux => ./third_party/webgpu/libs-linux
