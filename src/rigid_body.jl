export Pose6
struct Pose6{T}
    R::QuatRotation{T}
    t::SVector{3,T}
end
Adapt.@adapt_structure Pose6

Pose6(R, t::SVector{3,T}) where {T} = Pose6(QuatRotation{T}(R), t)
import Base.length, Base.broadcastable
length(::Pose6) = 0
Base.broadcastable(p::Pose6) = Ref(p)

import Base.:*, Base.one
*(p::Pose6, x::AbstractVector) = p.R * x + p.t
*(p::Pose6, span::Span) = p .* span
*(p1::Pose6, p2::Pose6) = Pose6(QuatRotation(p1.R * p2.R).q, p1.R * p2.t + p1.t)
one(::Union{Type{Pose6{T}},Pose6{T}}) where {T} = Pose6{T}(SMatrix{3,3,T,9}(I), SVector{3,T}(0, 0, 0))

struct InvPose6{T}
    pose::Pose6{T}
end
Adapt.@adapt_structure InvPose6

inv(p::Pose6) = InvPose6(p)
*(ip::InvPose6, x::AbstractVector) = ip.pose.R \ (x - ip.pose.t)

export Vel6
struct Vel6{T}
    ω::SVector{3,T}
    v::SVector{3,T}
end

import Base.:+
+(v1::Vel6, v2::Vel6) = Vel6(v1.ω + v2.ω, v1.v + v2.v)

import Base.zero
zero(::Union{Type{Vel6{T}},Vel6{T}}) where {T} = Vel6{T}(SVector{3,T}(0, 0, 0), SVector{3,T}(0, 0, 0))

@inline function integrate(p::Pose6, v::Vel6, dt)
    ωx = SA_F32[0 -v.ω[3] v.ω[2]; v.ω[3] 0 -v.ω[1]; -v.ω[2] v.ω[1] 0]
    dR = exp(ωx * dt)
    return Pose6{Float32}(dR * p.R, p.t + dt * v.v)
end

export Body
struct Body{T,GeomT,VizT}
    pose::Pose6{T}
    vel::Vel6{T}
    geom::GeomT
    viz::VizT
end
Body(; geom::G, viz::V) where {G,V} = Body{Float32,G,V}(one(Pose6{Float32}), zero(Vel6{Float32}), geom, viz)
Adapt.@adapt_structure Body

export Bodies
struct Bodies{
    PosesT<:AbstractVector,
    VelsT<:AbstractVector,
    GeomsT<:Tuple,
    VizsT<:Tuple
}
    poses::PosesT
    vels::VelsT
    geoms::GeomsT
    vizs::VizsT
end
Adapt.@adapt_structure Bodies

Bodies(bodies::Vararg{Body{T}}) where {T} = Bodies(
    Pose6{T}[body.pose for body in bodies],
    Vel6{T}[body.vel for body in bodies],
    Tuple(body.geom for body in bodies),
    Tuple(body.viz for body in bodies),
)

Bodies(bodies::Tuple) = Bodies(bodies...)

@inline function point_vel(pose::Pose6, vel::Vel6, x)
    return vel.ω × (x - pose.t) + vel.v
end
