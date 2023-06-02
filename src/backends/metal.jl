using .Metal

"Runs a kernel over nparticle threads."
@inline function run_kernel(kernel, ::MetalBackend, n, args...)
    function indexed_kernel(n, args...)
        i = Int32(thread_position_in_grid_1d())
        @inbounds if i <= n
            kernel(i, args...)
        end
        return nothing
    end

    kernel_compiled = @metal launch = false indexed_kernel(n, args...)
    # The pipeline state automatically computes occupancy stats 
    threads = min(n, kernel_compiled.pipeline_state.maxTotalThreadsPerThreadgroup)
    grid = cld(n, threads)

    kernel_compiled(n, args...; threads, grid)
end

"Sends host memory to the compute backend."
@inline function (::MetalBackend)(x)
    adapt(MtlArray, x)
end

"Replaces element a[i] with x, atomically. Returns previous value of a[i]."
@inline function atomic_replace!(::MetalBackend, a::AbstractArray{T}, i::Int32, x::T) where {T}
    atomic_replace!(SinglethreadedCPUBackend(), a, i, x)
end