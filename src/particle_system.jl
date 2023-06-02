export ParticleParams
Base.@kwdef struct ParticleParams{Tr,Tm,Tμ,Tγ,Tvd}
    r::Tr = 0.06
    m::Tm = 1.0
    μ::Tμ = 0.1
    γ::Tγ = 1e-2
    vd::Tvd = 0.9
end


export ParticleSystem
Base.@kwdef struct ParticleSystem{
    T,
    VT<:AbstractVector{T},
    BodiesT<:Bodies,
    ParamsT<:ParticleParams,
}
    x::VT
    v::VT
    bodies::BodiesT
    bodies_prev::BodiesT
    params::ParamsT
end
Adapt.@adapt_structure ParticleSystem