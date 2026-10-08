package wagogpu

//go:generate go run ./internal/genabi

import (
	"fmt"
	wago "github.com/wago-org/wago"
	"strings"
	"time"
)

type ElementType uint32
type BufferAccess uint8

const (
	AccessRead BufferAccess = iota + 1
	AccessWrite
	AccessReadWrite
)

type BindingConfig struct {
	Slot   uint32
	Type   ElementType
	Access BufferAccess
}
type KernelConfig struct {
	ID                uint32
	Export, CPUExport string
	Bindings          []BindingConfig
	RelaxedFloat      bool
	MinElements       uint32
}
type CompileErrorKind uint8

const (
	CompileInvalidContract CompileErrorKind = iota + 1
	CompileUnsupported
	CompileLimit
)

// CompileError is inspectable with errors.As, including after wrapping.
type CompileError struct {
	Kind    CompileErrorKind
	Message string
}

func (e *CompileError) Error() string { return e.Message }
func compileError(k CompileErrorKind, f string, a ...any) error {
	return &CompileError{k, fmt.Sprintf(f, a...)}
}

type elementSpec struct {
	suffix string
	size   uint64
	scalar wago.ValType
}

func (t ElementType) spec() elementSpec {
	if t == 0 || int(t) >= len(elementSpecs) {
		return elementSpec{}
	}
	return elementSpecs[t]
}

type bufferImport struct {
	name            string
	params, results []wago.ValType
}

func validateBufferConfig(c *Config) error {
	defaults := []struct {
		v   *uint32
		max uint32
	}{{&c.MaxBuffersPerInstance, 64}, {&c.MaxKernelsPerModule, 64}, {&c.MaxBindingsPerKernel, 8}}
	for _, d := range defaults {
		if *d.v == 0 {
			*d.v = d.max
		}
		if *d.v > d.max {
			return fmt.Errorf("buffer configuration limit exceeds %d", d.max)
		}
	}
	for _, d := range []struct {
		v   *uint64
		max uint64
	}{{&c.MaxInstanceBufferBytes, 256 << 20}, {&c.MaxRuntimeBufferBytes, 512 << 20}} {
		if *d.v == 0 {
			*d.v = d.max
		}
		if *d.v > d.max {
			return fmt.Errorf("buffer memory limit exceeds %d", d.max)
		}
	}
	if uint32(len(c.Kernels)) > c.MaxKernelsPerModule {
		return fmt.Errorf("too many kernels")
	}
	ids := map[uint32]bool{}
	names := map[string]bool{}
	c.Kernels = append([]KernelConfig(nil), c.Kernels...)
	for i := range c.Kernels {
		k := &c.Kernels[i]
		if err := validateKernel(*k, c.MaxBindingsPerKernel); err != nil {
			return err
		}
		if ids[k.ID] || names[k.Export] || names[k.CPUExport] {
			return fmt.Errorf("duplicate kernel ID or export")
		}
		ids[k.ID] = true
		names[k.Export] = true
		names[k.CPUExport] = true
		if k.MinElements > c.MaxElements {
			return fmt.Errorf("kernel threshold exceeds element limit")
		}
		k.Bindings = append([]BindingConfig(nil), k.Bindings...)
	}
	return nil
}
func validateKernel(k KernelConfig, max uint32) error {
	suffix := strings.TrimPrefix(k.Export, "wago_gpu.kernel.")
	if k.ID == 0 || suffix == "" || suffix == k.Export || k.CPUExport != "wago_gpu.cpu."+suffix {
		return compileError(CompileInvalidContract, "invalid kernel ID or export names")
	}
	if len(k.Export) > 256 || len(k.CPUExport) > 256 || len(k.Bindings) == 0 || uint32(len(k.Bindings)) > max {
		return compileError(CompileInvalidContract, "invalid kernel binding or name limit")
	}
	var seen [8]bool
	for _, b := range k.Bindings {
		if b.Slot >= 8 || seen[b.Slot] || b.Type.spec().size == 0 || b.Access < AccessRead || b.Access > AccessReadWrite {
			return compileError(CompileInvalidContract, "invalid slot, type, or access")
		}
		seen[b.Slot] = true
	}
	return nil
}

type BufferSnapshot struct {
	PeakRuntimeBufferBytes                                     uint64
	State, Reason, DeviceState                                 string
	Last                                                       BufferOperation
	Totals                                                     TransferCounters
	TotalsComplete                                             bool
	Kernels                                                    []KernelDiagnostic
	Instances                                                  []InstanceDiagnostic
	PipelineHits, PipelineMisses                               uint64
	PendingCompilations, PendingContracts, PendingTranslations uint32
	PendingContractBytes, PendingWGSLBytes                     uint64
	ActiveCompilations                                         uint32
	RuntimeBufferBytes, RuntimePoolBytes, LegacyBufferBytes    uint64
}
type BufferOperation struct {
	Sequence                                                                                   uint64
	Instance                                                                                   wago.InstanceIdentity
	KernelID                                                                                   uint32
	Import                                                                                     string
	Status                                                                                     int32
	ReasonCode, Reason, Outcome                                                                string
	Count                                                                                      uint32
	HardwareSubmitted, TransferCountsComplete, ProfileValid, TimestampsValid                   bool
	QueueWait, Total, Upload, DeviceCopy, Compute, Download, Conversion, Commit, DeviceCompute time.Duration
	TransferCounters
}
type TransferCounters struct {
	GuestSetBytes, GuestCopyBytes, GuestSetCount, GuestCopyCount                                           uint64
	LogicalUploadBytes, LogicalDownloadBytes, GPUUploadBytes, GPUDownloadBytes, UploadCount, DownloadCount uint64
	DeviceCopyBytes, DeviceCopyCount, ParameterUploadBytes, ParameterUploadCount                           uint64
}
type KernelDiagnostic struct {
	Translate, Pipeline time.Duration
	Module              wago.ModuleIdentity
	KernelID            uint32
	State, Reason       string
	ContractVerified    bool
}
type InstanceDiagnostic struct {
	PeakBytes                                                                                      uint64
	Instance                                                                                       wago.InstanceIdentity
	Buffers, CPUCurrent, GPUCurrent, BothCurrent, ContentsLost                                     uint32
	LogicalBytes, CPUBytes, GPUBytes, StagingBytes, ScratchBytes, RetainedPoolBytes, ReservedBytes uint64
}
