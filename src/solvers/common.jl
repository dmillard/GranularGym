@inline static_col(m::AbstractMatrix, i) = @inbounds SA_F32[m[1, i], m[2, i], m[3, i]]

@inline function orthogonal_frame(v::AbstractVector)
    # https://graphics.pixar.com/library/OrthonormalB/paper.pdf
    nv = normalize(v)
    x, y, z = nv
    sign = copysign(1.0f0, z)
    a = -1.0f0 / (sign + z)
    b = x * y * a
    t1 = SA_F32[1.0f0+sign*x*x*a, sign*b, -sign*x]
    t2 = SA_F32[b, sign+y*y*a, -y]
    return hcat(nv, t1, t2)
end


@inline function project_circle(v::AbstractVector, r)
    l = norm(v)
    return l <= r ? v : v * (r / l)
end

function find_neighbors(system, radius)
    kdtree = KDTree(system.x)
    idxs = inrange(kdtree, system.x, radius)
    return idxs
end

struct ContactPair{T}
    i::Int32
    j::Int32
    ψ::T
    n::SVector{3,T}
end
Adapt.@adapt_structure ContactPair

function find_contact_pairs(system::ParticleSystem{T}, neighbors) where {T}
    contactpairs = ContactPair{T}[]
    for (i, ineighbors) ∈ enumerate(neighbors)
        for j ∈ ineighbors
            if i == j
                continue
            end
            xi = SVector{3,T}(@views system.x[:, i])
            xj = SVector{3,T}(@views system.x[:, j])
            n = xi - xj
            nn = norm(n)
            if (nn <= 2 * system.params.r)
                ψ = nn - 2 * system.params.r
                push!(contactpairs, ContactPair(i, j, ψ, n))
            end
        end
    end

    for (i, x) ∈ enumerate(eachcol(system.x)), body ∈ system.bodies
        ψ = body.geometry(x) - system.params.r
        if ψ <= 0
            push!(contactpairs, ContactPair(i, -1, ψ, normgrad(body.geometry, x)))
        end
    end

    return contactpairs
end

function contact_jacobian(system, contactpairs)
    dim, nparticles = size(system.x)
    ncontacts = length(contactpairs)

    JI, JJ, JV = zeros(Int, 9 * ncontacts), zeros(Int, 9 * ncontacts), zeros(9 * ncontacts)
    for (ci, contact) ∈ enumerate(contactpairs)
        Ji = orthogonal_frame(contact.n)'
        abi = 1
        for b ∈ 1:3, a ∈ 1:3
            JI[9*(ci-1)+abi] = 3 * (ci - 1) + a
            JJ[9*(ci-1)+abi] = 3 * (contact.i - 1) + b
            JV[9*(ci-1)+abi] = Ji[a, b]
            abi += 1
        end
    end
    J = sparse(JI, JJ, JV, dim * ncontacts, dim * nparticles)
    return J
end