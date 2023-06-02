using .CUDA
CUDA.allowscalar(false)

"Runs a kernel over nparticle threads."
@inline function run_kernel(kernel, ::CUDABackend, n, args...)
    function indexed_kernel(n, args...)
        i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        @inbounds if i <= n
            kernel(i, args...)
        end
        return nothing
    end

    kernel_compiled = @cuda launch = false indexed_kernel(n, args...)
    config = launch_configuration(kernel_compiled.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)

    kernel_compiled(n, args...; threads, blocks)
end

"Sends host memory to the compute backend."
@inline function (::CUDABackend)(x)
    cu(x)
end

"Replaces element a[i] with x, atomically. Returns previous value of a[i]."
@inline function atomic_replace!(::CUDABackend, a::AbstractArray{T}, i::Int32, x::T) where {T}
    p = pointer(a, i)
    old = a[i]
    while a[i] === old
        old = CUDA.atomic_cas!(p, old, x)
    end
    return old
end