module github.com/oliverbestmann/webgpu

go 1.25

require (
	github.com/oliverbestmann/webgpu/libs-linux v0.0.0-20260628152803-421b8a341d08
	github.com/stretchr/testify v1.11.1
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

retract v1.27.0 // published before deciding on a version scheme. we start at v1.0.0
