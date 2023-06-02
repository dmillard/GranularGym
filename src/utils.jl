@inline sqnorm(x) = dot(x, x)

@inline function normgrad(f, x::AbstractVector)
    h = 1e-4
    grad = SA_F32[
        f(x + SA_F32[h, 0, 0])-f(x - SA_F32[h, 0, 0]),
        f(x + SA_F32[0, h, 0])-f(x - SA_F32[0, h, 0]),
        f(x + SA_F32[0, 0, h])-f(x - SA_F32[0, 0, h]),
    ]
    return normalize(grad)
end

const Span = Tuple{SVector{3,Float32},SVector{3,Float32}}
centered_span(width::SVector{3,Float32}) = (-width / 2, width / 2)
span_width((lo, hi)::Span) = hi - lo

function bounding_span(spans, poses)
    lo = SA_F32[Inf, Inf, Inf]
    hi = SA_F32[-Inf, -Inf, -Inf]
    for (span, pose) ∈ zip(spans, poses)
        x1, x2 = pose .* span
        lo = min.(lo, x1, x2)
        hi = max.(hi, x1, x2)
    end
    return (lo, hi)
end

function bounding_span(points::AbstractVector{<:SVector{3}})
    lo = SA_F32[Inf, Inf, Inf]
    hi = SA_F32[-Inf, -Inf, -Inf]
    for p ∈ points
        if (all(isfinite.(p)))
            lo = min.(lo, p)
            hi = max.(hi, p)
        end
    end
    return (lo, hi)
end

export uniform_grid_point
function uniform_grid_point(span::Span, resolution::Real, index::Integer)
    lo, hi = span
    ns = Int32.(floor.((hi - lo) ./ resolution))
    if index > prod(ns)
        @warn "Can't generate more than $(ns[1]) x $(ns[2]) x $(ns[3]) = $(prod(ns)) points"
    end
    is = SA{Int32}[
        index%ns[1],
        (index÷ns[1])%ns[2],
        (index÷(ns[1]*ns[2]))%ns[3],
    ]
    x = Float32.(is .* resolution .+ lo)
    return x
end

@inline function slerp(q1::QuatRotation, q2::QuatRotation, t)
    # http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/
    iszero(q1) && throw(DomainError(q2, "The input quaternion must be non-zero."))
    iszero(q2) && throw(DomainError(q2, "The input quaternion must be non-zero."))
    coshalftheta = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z

    if coshalftheta < 0
        q2 = QuatRotation(-q2.w, -q2.x, -q2.y, -q2.z)
        coshalftheta = -coshalftheta
    end

    if coshalftheta < 1
        halftheta = acos(coshalftheta)
        sinhalftheta = sqrt(1 - coshalftheta^2)

        ratio_1 = sin((1 - t) * halftheta) / sinhalftheta
        ratio_2 = sin(t * halftheta) / sinhalftheta
    else
        ratio_1 = float(1 - t)
        ratio_2 = float(t)
    end

    return QuatRotation(
        q1.w * ratio_1 + q2.w * ratio_2,
        q1.x * ratio_1 + q2.x * ratio_2,
        q1.y * ratio_1 + q2.y * ratio_2,
        q1.z * ratio_1 + q2.z * ratio_2,
        true
    )
end