abstract type AbstractComputeBackend end

export SinglethreadedCPUBackend
struct SinglethreadedCPUBackend <: AbstractComputeBackend end
export MultithreadedCPUBackend
struct MultithreadedCPUBackend <: AbstractComputeBackend end
include("./backends/cpu.jl")

export CUDABackend
struct CUDABackend <: AbstractComputeBackend
    function CUDABackend()
        @assert isdefined(Main, :CUDA)
        new()
    end
end
push!(INIT_FUNCTIONS, () -> begin
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("./backends/cuda.jl")
end)

export MetalBackend
struct MetalBackend <: AbstractComputeBackend
    function MetalBackend()
        @assert isdefined(Main, :Metal)
        new()
    end
end
push!(INIT_FUNCTIONS, () -> begin
    @require Metal = "dde4c033-4e86-420c-a63e-0dd931031962" include("./backends/metal.jl")
end)