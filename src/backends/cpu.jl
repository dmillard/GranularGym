"Runs a kernel over nparticle threads."
@inline function run_kernel(kernel, ::SinglethreadedCPUBackend, n, args...)
    for i ∈ Int32.(1:n)
        kernel(i, args...)
    end
end

"Sends host memory to the compute backend."
@inline function (::SinglethreadedCPUBackend)(x)
    identity(x)
end

"Replaces element a[i] with x, atomically. Returns previous value of a[i]."
@inline function atomic_replace!(::SinglethreadedCPUBackend, a::AbstractArray{T}, i::Int32, x::T) where {T}
    old = a[i]
    a[i] = x
    return old
end

"Runs a kernel over nparticle threads."
@inline function run_kernel(kernel, ::MultithreadedCPUBackend, n, args...)
    @threads for i ∈ Int32.(1:n)
        kernel(i, args...)
    end
end

"Sends host memory to the compute backend."
@inline function (::MultithreadedCPUBackend)(x)
    identity(x)
end

"Replaces element a[i] with x, atomically. Returns previous value of a[i]."
@inline function atomic_replace!(::MultithreadedCPUBackend, a::AbstractArray{T}, i::Int32, x::T) where {T}
    atomic_replace!(SinglethreadedCPUBackend(), a, i, x)
end