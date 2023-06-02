export SymplecticEuler
struct SymplecticEuler{T,VT<:AbstractVector{SVector{3,T}}}
    dt::T
    impulse::VT
end
Adapt.@adapt_structure SymplecticEuler

SymplecticEuler(system::ParticleSystem, dt::Real) = SymplecticEuler(dt, zero(system.v))

export step!
function step!(
    backend::AbstractComputeBackend,
    system::ParticleSystem,
    solver::AbstractContactSolver,
    stepper::SymplecticEuler,
    f_ext::AbstractVector,
    t::Real;
)
    solve!(backend, stepper.impulse, solver, system, f_ext, stepper.dt, t)

    #system.bodies.poses .= integrate.(system.bodies.poses, system.bodies.vels, stepper.dt)

    system.v .+= (stepper.impulse .+ stepper.dt .* f_ext) ./ system.params.m
    system.v .*= exp(log(1 - system.params.γ) * stepper.dt)
    system.x .+= stepper.dt .* system.v

    # Global speed limit, revisit
    map(system.v) do vi
        clamp.(vi, -50, 50)
    end
end