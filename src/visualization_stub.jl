export Visualizer
struct Visualizer{RealVizT}
    vis::RealVizT
end

function Visualizer(args...; kw...)
    if !isdefined(Main, :GLMakie)
        error("Visualization requires GLMakie to be loaded!")
    else
        # May as well do this here
        @eval GLMakie.GLAbstraction begin
            # XXX: to make scalar iteration error
            function gpu_getindex(b::GLBuffer{T}, range::UnitRange) where {T}
                error("GLBuffer getindex")  # XXX: for development
                multiplicator = sizeof(T)
                offset = first(range) - 1
                value = Vector{T}(undef, length(range))
                GLMakie.bind(b)
                GLMakie.GLAbstraction.glGetBufferSubData(b.buffertype, multiplicator * offset, sizeof(value), value)
                GLMakie.bind(b, 0)
                return value
            end
        end

        Visualizer(Visualizer_(args...; kw...))
    end
end

(v::Visualizer)(args...; kw...) = v.vis(args...; kw...)