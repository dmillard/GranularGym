struct LinkedHashSet{
    IntsT<:AbstractVector{Int32},
    SVector3IntsT<:AbstractVector{SVector{3,Int32}}
}
    hash_table::IntsT
    cell_ids::SVector3IntsT
    nexts::IntsT
end
Adapt.@adapt_structure LinkedHashSet

LinkedHashSet(n::Int32, hash_table_size) = LinkedHashSet(
    zeros(Int32, hash_table_size),
    zeros(SVector{3,Int32}, n),
    zeros(Int32, n)
)

@enum InterparticleAlgorithm begin
    oneloop
    twoloops_fused
    twoloops_separate
end

export ProjectedJacobi
struct ProjectedJacobi{
    LinkedHashSetT<:LinkedHashSet,
    NWT<:AbstractMatrix{Int32}
} <: AbstractContactSolver
    nsteps::Int32
    interparticle_algorithm::InterparticleAlgorithm
    hash_set::LinkedHashSetT
    neighbor_workspace::NWT
end
Adapt.@adapt_structure ProjectedJacobi

function ProjectedJacobi(;
    nsteps::Int32,
    nparticles::Int32,
    interparticle_algorithm::InterparticleAlgorithm=twoloops_separate,
    hash_table_size=nothing
)
    if hash_table_size === nothing
        hash_table_size = nextprime(nparticles * 2)
    end
    return ProjectedJacobi(
        nsteps,
        interparticle_algorithm,
        LinkedHashSet(nparticles, hash_table_size),
        zeros(Int32, (512, nparticles))
    )
end

function spatial_hash(xi::AbstractVector{Int32}, n::Int32)
    @inbounds begin
        o = Int32(100)
        # https://matthias-research.github.io/pages/publications/tetraederCollision.pdf
        return (foldl(xor, (
            (xi[1] - o) * Int32(73856093),
            (xi[2] - o) * Int32(19349663),
            (xi[3] - o) * Int32(83492791)
        )) % n + n) % n + Int32(1)
    end
end

function neighbor_hashes(xi::AbstractVector, n::Int32)
    return (
        spatial_hash(xi + SA[o1, o2, o3], n)
        for o1 ∈ Int32(-1):Int32(1), o2 ∈ Int32(-1):Int32(1), o3 ∈ Int32(-1):Int32(1)
    )
end

@inline function apply_contact!(
    impulse::AbstractVector{SVector{3,Float32}},
    v::AbstractVector{SVector{3,T}},
    params::ParticleParams,
    f_ext::AbstractVector{SVector{3,Float32}},
    dt::Float32,
    contact::ContactPair,
    vj_rigid::SVector{3,T}=zero(SVector{3,T})
) where {T}
    @inbounds begin
        i = contact.i
        j = contact.j

        impi = impulse[i] # Initial impulse
        vi = v[i]
        fi = f_ext[i]
        vj = if j > 0 # Neighbor is a particle
            damp = 1 - params.vd
            damp * v[j]
        else # Neighbor is a rigid body
            vj_rigid
        end
        n = if all(contact.n .== 0)
            SA_F32[1, 0, 0]
        else
            contact.n
        end
        Ji = orthogonal_frame(n)

        # Compute global frame b residual in NCP
        bg = vi - vj
        bg = bg + dt * fi / params.m
        bg = bg + impi

        bln, blt1, blt2 = Ji' * bg # Transform into contact frame
        bln += 0.02f0 * contact.ψ / dt # Baumgarte stabilization

        # Local change in impulse
        dlimpin = max(-bln, 0)
        dlimpit = project_circle(SA_F32[-blt1, -blt2], params.μ * dlimpin)

        # Apply global impulse change
        dgimpi = Ji * SA_F32[dlimpin, dlimpit[1], dlimpit[2]]

        impulse[i] += dgimpi
    end
    return nothing
end

@inline function find_and_apply_interparticle_contacts_oneloop!(
    particle_i::Int32,
    impulse::AbstractVector{SVector{3,Float32}},
    solver::ProjectedJacobi,
    x::AbstractVector{SVector{3,Float32}},
    v::AbstractVector{SVector{3,Float32}},
    params::ParticleParams,
    f_ext::AbstractVector{SVector{3,Float32}},
    dt::Float32,
)
    hash_table_size = Int32(length(solver.hash_set.hash_table))
    @inbounds begin
        cell_id = solver.hash_set.cell_ids[particle_i]
        xi = x[particle_i]
        # Iterate linked hashset of neighbors
        for hj ∈ neighbor_hashes(cell_id, hash_table_size)
            j = solver.hash_set.hash_table[hj]
            while j != 0
                xj = x[j]
                n = xi - xj # Contact normal
                nnsq = sqnorm(n)
                # Check collisions and apply contacts
                if particle_i != j && nnsq <= (2 * params.r)^2
                    ψ = Float32(sqrt(nnsq) - 2 * params.r)
                    contact = ContactPair(particle_i, j, ψ, n)
                    apply_contact!(impulse, v, params, f_ext, dt, contact)
                    contact = ContactPair(j, particle_i, ψ, -n)
                    apply_contact!(impulse, v, params, f_ext, dt, contact)
                end
                j = solver.hash_set.nexts[j]
            end
        end
    end

    return nothing
end

@inline function find_and_apply_interparticle_contacts_twoloops_fused!(
    particle_i::Int32,
    impulse::AbstractVector{SVector{3,Float32}},
    solver::ProjectedJacobi,
    x::AbstractVector{SVector{3,Float32}},
    v::AbstractVector{SVector{3,Float32}},
    params::ParticleParams,
    f_ext::AbstractVector{SVector{3,Float32}},
    dt::Float32,
)
    find_interparticle_collisions!(solver, x, params, particle_i)
    apply_interparticle_contacts_only!(impulse, solver, x, v, params, f_ext, dt, particle_i)

    return nothing
end

@inline function find_interparticle_collisions!(
    particle_i::Int32,
    solver::ProjectedJacobi,
    x::AbstractVector,
    params::ParticleParams,
)
    hash_table_size = Int32(length(solver.hash_set.hash_table))
    @inbounds begin
        cell_id = solver.hash_set.cell_ids[particle_i]
        xi = x[particle_i]
        # Iterate linked hashset of neighbors
        contact_count = 0
        for hj ∈ neighbor_hashes(cell_id, hash_table_size)
            j = solver.hash_set.hash_table[hj]
            while j != 0
                xj = x[j]
                n = xi - xj # Contact normal
                nnsq = sqnorm(n)
                # Check collisions and apply contacts
                if particle_i != j && nnsq <= (2 * params.r)^2
                    contact_count += 1
                    solver.neighbor_workspace[contact_count, particle_i] = j
                end
                j = solver.hash_set.nexts[j]
            end
        end

        solver.neighbor_workspace[contact_count+1, particle_i] = 0
    end

    return nothing
end

@inline function apply_interparticle_contacts_only!(
    particle_i::Int32,
    impulse::AbstractVector{SVector{3,Float32}},
    solver::ProjectedJacobi,
    x::AbstractVector,
    v::AbstractVector,
    params::ParticleParams,
    f_ext::AbstractVector{SVector{3,Float32}},
    dt::Float32,
)
    @inbounds begin
        xi = x[particle_i]
        # Iterate linked hashset of neighbors
        for contact_i ∈ 1:size(solver.neighbor_workspace, 1)
            j = solver.neighbor_workspace[contact_i, particle_i]
            if j == 0
                # Sentry value for end of contacts
                break
            end
            xj = x[j]
            n = xi - xj # Contact normal
            nnsq = sqnorm(n)

            ψ = Float32(sqrt(nnsq) - 2 * params.r)
            contact = ContactPair(particle_i, j, ψ, n)
            apply_contact!(impulse, v, params, f_ext, dt, contact)
            contact = ContactPair(j, particle_i, ψ, -n)
            apply_contact!(impulse, v, params, f_ext, dt, contact)
        end
    end

    return nothing
end

@inline function find_and_apply_rigid_body_contacts!(
    particle_i::Int32,
    impulse::AbstractVector{SVector{3,Float32}},
    solver::ProjectedJacobi,
    x::AbstractVector,
    v::AbstractVector,
    params::ParticleParams,
    f_ext::AbstractVector{SVector{3,Float32}},
    dt::Float32,
    body_geom::GeometrySDF,
    body_pose::Pose6,
    body_pose_prev::Pose6,
)
    @inbounds begin
        xgi = x[particle_i] # global coords
        xli = inv(body_pose) * xgi # local coords
        ψ = body_geom(xli) - params.r
        if ψ <= 0
            ng = normgrad(xgi -> body_geom(inv(body_pose) * xgi), xgi) # global normal
            contact = ContactPair(particle_i, Int32(-1), ψ, ng)
            xgi_delta = body_pose_prev * xli
            vel_rigid = (xgi - xgi_delta) / 1.0f-6 # TODO factor this out properly
            apply_contact!(impulse, v, params, f_ext, dt, contact, vel_rigid)
        end
    end
    return nothing
end

"Atomically creates a spatial hashtable from a matrix of particle positions."
@inline function solve_hashtable!(
    i::Int32,
    backend::AbstractComputeBackend,
    solver::ProjectedJacobi,
    x::AbstractVector{SVector{3,Float32}},
    params::ParticleParams,
)
    @inbounds begin
        cell_size = 2.01f0 * params.r
        hash_table_size = Int32(length(solver.hash_set.hash_table))

        cell_id = unsafe_trunc.(Int32, round.(x[i] ./ cell_size))
        solver.hash_set.cell_ids[i] = cell_id

        # Build linked hashtable
        h = spatial_hash(cell_id, hash_table_size)
        new_head = Int32(i)
        current_head = atomic_replace!(backend, solver.hash_set.hash_table, h, new_head)
        solver.hash_set.nexts[i] = current_head
    end
    return nothing
end

"Solves for the contact impulse of the system given the system state and velocity."
@inline function solve!(
    backend::AbstractComputeBackend,
    impulse::AbstractVector,
    solver::ProjectedJacobi,
    system::ParticleSystem,
    f_ext::AbstractVector,
    dt::Real,
    t::Real
)
    nparticles = length(impulse)
    impulse .= Ref(zero(eltype(impulse)))
    solver.hash_set.hash_table .= 0
    solver.hash_set.nexts .= 0

    # Build cell-id hashtable
    run_kernel(
        solve_hashtable!,
        backend,
        nparticles,
        backend,
        solver,
        system.x,
        system.params
    )
    if solver.interparticle_algorithm === twoloops_separate
        run_kernel(
            find_interparticle_collisions!,
            backend,
            nparticles,
            solver,
            system.x,
            system.params
        )
        for _ ∈ 1:solver.nsteps
            run_kernel(
                apply_interparticle_contacts_only!,
                backend,
                nparticles,
                impulse,
                solver,
                system.x,
                system.v,
                system.params,
                f_ext,
                dt
            )
        end
    elseif solver.interparticle_algorithm === twoloops_fused
        for _ ∈ 1:solver.nsteps
            run_kernel(
                find_and_apply_interparticle_contacts_twoloops_fused!,
                backend,
                nparticles,
                impulse,
                solver,
                system.x,
                system.v,
                system.params,
                f_ext,
                dt
            )
        end
    elseif solver.interparticle_algorithm === oneloop
        for _ ∈ 1:solver.nsteps
            run_kernel(
                find_and_apply_interparticle_contacts_oneloop!,
                backend,
                nparticles,
                impulse,
                solver,
                system.x,
                system.v,
                system.params,
                f_ext,
                dt
            )
        end
    end
    for body_i ∈ 1:length(system.bodies.geoms)
        begin # TODO: remove
            run_kernel(
                find_and_apply_rigid_body_contacts!,
                backend,
                nparticles,
                impulse,
                solver,
                system.x,
                system.v,
                system.params,
                f_ext,
                dt,
                system.bodies.geoms[body_i],
                system.bodies.poses[body_i],
                system.bodies_prev.poses[body_i], # TODO: Add to bodies?
            )
        end
    end

    return impulse
end