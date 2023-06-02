export GeometrySDF
abstract type GeometrySDF{T} end

export CompositeGeometrySDF
Base.@kwdef struct CompositeGeometrySDF{T,GeomsT} <: GeometrySDF{T}
    pose::Pose6{T}
    span::Span
    subgeometries::GeomsT
end
Adapt.@adapt_structure CompositeGeometrySDF

function CompositeGeometrySDF(pose::Pose6, subgeometries)
    span = bounding_span(
        (sg.span for sg ∈ subgeometries),
        (sg.pose for sg ∈ subgeometries),
    )
    return CompositeGeometrySDF(pose, span, subgeometries)
end

function (g::CompositeGeometrySDF)(x::AbstractVector)
    xrel = inv(g.pose) * x
    return minimum(g.subgeometries) do subgeometry
        subgeometry(xrel)
    end
end

export BoxSDF
struct BoxSDF{T} <: GeometrySDF{T}
    pose::Pose6{T}
    span::Span
    dscale::T # Rescale SDF (not the BoxSDF itself)
end
BoxSDF(; span, dscale=1.0f0) = BoxSDF{Float32}(one(Pose6{Float32}), span, dscale)
Adapt.@adapt_structure BoxSDF

@inline function (g::BoxSDF{T})(x::AbstractVector) where {T}
    lo, hi = g.span
    xrel = inv(g.pose) * x
    q = abs.(xrel - (hi + lo) / 2) - (hi - lo) / 2
    return g.dscale * (norm(max.(q, 0)) + min(0, maximum(q)))
end

export CylinderSDF
Base.@kwdef struct CylinderSDF{T} <: GeometrySDF{T}
    pose::Pose6{T}
    span::Span
    cap1::SVector{3,T}
    cap2::SVector{3,T}
    radius::T
    dscale::T = one(T) # Rescale SDF (not the BoxSDF itself)
end

function CylinderSDF(
    pose::Pose6{T},
    cap1::SVector{3,T},
    cap2::SVector{3,T},
    radius::T,
    dscale::T=one(T)
) where {T}
    lo, hi = bounding_span([cap1, cap2])
    span = (lo .- radius, hi .+ radius)
    return CylinderSDF(pose, span, cap1, cap2, radius, dscale)
end

function (g::CylinderSDF{T})(x::AbstractVector) where {T}
    # https://iquilezles.org/articles/distfunctions/
    a, b = g.cap1, g.cap2
    ba = b - a
    xa = x - a
    baba = dot(ba, ba)
    xaba = dot(xa, ba)
    x = norm(xa * baba - ba * xaba) - g.radius * baba
    y = abs(xaba - baba * 0.5f0) - baba * 0.5f0
    x2 = x * x
    y2 = y * y * baba
    d = (max(x, y) < zero(T)) ? -min(x2, y2) : (((x > zero(T)) ? x2 : zero(T)) + ((y > zero(T)) ? y2 : zero(T)))
    return g.dscale * sign(d) * sqrt(abs(d)) / baba
end

@inline function min_dist(poses::AbstractVector{<:Pose6}, geoms, x)
    return minimum(zip(poses, geoms)) do (pose, geom)
        geom(inv(pose) * x)
    end
end

MeshVertex = SVector{3,Float32}
MeshTri = SVector{3,MeshVertex}

# https://iquilezles.org/articles/distfunctions/
@inline function unsigned_dist_tri(tri::MeshTri, x::MeshVertex)
    @inbounds begin
        a, b, c = tri
        ba = b - a
        xa = x - a
        cb = c - b
        xb = x - b
        ac = a - c
        xc = x - c
        nor = cross(ba, ac)

        return sqrt(
            (
                sign(dot(cross(ba, nor), xa)) +
                sign(dot(cross(cb, nor), xb)) +
                sign(dot(cross(ac, nor), xc)) < 2.0f0
            )
            ?
            min(
                min(
                    sqnorm(ba * clamp(dot(ba, xa) / sqnorm(ba), 0.0f0, 1.0f0) - xa),
                    sqnorm(cb * clamp(dot(cb, xb) / sqnorm(cb), 0.0f0, 1.0f0) - xb)
                ),
                sqnorm(ac * clamp(dot(ac, xc) / sqnorm(ac), 0.0f0, 1.0f0) - xc)
            )
            :
            dot(nor, xa) * dot(nor, xa) / sqnorm(nor)
        )
    end
end

function compute_face_normal(vertices::MeshTri, normals::MeshTri)
    @inbounds begin
        face_normal_dir = cross(vertices[2] - vertices[1], vertices[3] - vertices[1])
        ds = dot.(Ref(face_normal_dir), normals)
        return normalize(face_normal_dir) * sign(sum(ds))
    end
end

function shrink_tri(vertices::MeshTri; abs_shrink_dist::Float32=1.0f-5)
    @inbounds begin
        barycenter = sum(vertices) / 3
        return map(vertices) do v
            shrink_dir = normalize(barycenter - v)
            return v + abs_shrink_dist * shrink_dir
        end
    end
end

function standoff_tri(vertices::MeshTri, normals::MeshTri; abs_standoff_dist::Float32=1.0f-5)
    @inbounds begin
        standoff = abs_standoff_dist * compute_face_normal(vertices, normals)
        return map(vertices) do v
            v + standoff
        end
    end
end

function moeller_trumbore_ray_triangle_intersection(
    tri::MeshTri,
    origin::SVector{3,Float32},
    dir::SVector{3,Float32}
)
    NO_INTERSECTION = (false, SA_F32[0, 0, 0])
    eps = 1.0f-8
    v1, v2, v3 = tri
    e1, e2 = v2 - v1, v3 - v1
    h = dir × e2
    a = dot(e1, h)
    if abs(a) < eps
        return NO_INTERSECTION
    end

    f = 1.0f0 / a
    s = origin - v1
    u = f * dot(s, h)
    if u < 0 || u > 1
        return NO_INTERSECTION
    end

    q = s × e1
    v = f * dot(dir, q)
    if v < 0 || u + v > 1
        return NO_INTERSECTION
    end

    t = f * dot(e2, q)
    if t < eps
        return NO_INTERSECTION
    end

    return (true, origin + t * dir)
end

function transform_tris(
    pose::Pose6,
    scale::Real,
    tri_vertices::Vector{MeshTri},
    tri_vertex_normals::Vector{MeshTri}
)
    @inbounds begin
        transformed_tri_vertices = map(tri_vertices) do tri
            map(tri) do v
                pose * v * scale
            end
        end
        transformed_tri_vertex_normals = map(tri_vertex_normals) do tri
            map(tri) do n
                pose.R * n
            end
        end
        return transformed_tri_vertices, transformed_tri_vertex_normals
    end
end

export MeshSDF
struct MeshSDF{
    T,
    VtxArrayT<:AbstractVector{MeshTri},
    NrmArrayT<:AbstractVector{MeshTri}
} <: GeometrySDF{T}
    pose::Pose6{T}
    span::Span
    scale::T
    tri_vertices::VtxArrayT
    tri_vertex_normals::NrmArrayT
end
Adapt.Adapt.@adapt_structure MeshSDF

function MeshSDF(filename::String; pose=one(Pose6{Float32}), scale=1.0f0)
    m = load(filename)

    function svecify_per_face(attribute)
        out = Vector{MeshTri}(undef, size(GeometryBasics.faces(m)))
        return map!(out, GeometryBasics.faces(m)) do f
            MeshTri(MeshVertex(attribute(m)[fi]) for fi in f)
        end
    end

    tri_vertices = svecify_per_face(GeometryBasics.coordinates)
    tri_vertex_normals = svecify_per_face(GeometryBasics.normals)
    tri_vertices, tri_vertex_normals = transform_tris(
        pose,
        scale,
        tri_vertices,
        tri_vertex_normals
    )
    tri_vertices = standoff_tri.(tri_vertices, tri_vertex_normals)

    span = bounding_span(reduce(vcat, Array.(tri_vertices)))
    return MeshSDF(pose, span, scale, tri_vertices, tri_vertex_normals)
end

function (sdf::MeshSDF)(x::SVector{3,Float32})
    xlocal = inv(sdf.pose) * x
    up = SA_F32[0, 0, 1]
    @inbounds begin
        mindist::Float32 = Inf
        minidx::Int = 0
        mindistsign::Float32 = 0.0
        for (i, (vs_i, ns_i)) ∈ enumerate(zip(sdf.tri_vertices, sdf.tri_vertex_normals))
            dist = unsigned_dist_tri(vs_i, xlocal)
            if isfinite(dist) && dist < mindist
                mindist = dist
                minidx = i
                d = dot(
                    xlocal - vs_i[1],
                    compute_face_normal(vs_i, ns_i)
                )
                mindistsign = sign(d)
            end
        end
        if mindist < 1.0f-5
            mindist = 0
        end
        return mindistsign * mindist
    end
end

function grid_ranges(span, resolution::Float32)
    lo = -resolution * (2 .+ ceil.(-span[1] ./ resolution))
    hi = resolution * (2 .+ ceil.(span[2] ./ resolution))
    return Tuple(range(; start=lo[i], step=resolution, stop=hi[i]) for i ∈ 1:3)
end

export GriddedSDF
struct GriddedSDF{
    T,
    ValueArrayT<:AbstractArray{T,3},
    ValueInterpolatorT
} <: GeometrySDF{T}
    pose::Pose6{T} # Pose of point [0, 0, 0] in span
    span::Span # x, y, z spans (not symmetric about pose)
    real_span::Span # Span arising from discretization
    resolution::T # Resolution
    values::ValueArrayT
    value_interpolator::ValueInterpolatorT
end
Adapt.@adapt_structure GriddedSDF

function GriddedSDF(
    backend::AbstractComputeBackend,
    sdf_pose::Pose6,
    span::Span,
    resolution::Real,
    value_fn,
    sample_origin_pose::Pose6=one(Pose6{Float32})
)
    x1s, x2s, x3s = grid_ranges(span, resolution)
    real_span = (
        SA_F32[first.((x1s, x2s, x3s))...],
        SA_F32[last.((x1s, x2s, x3s))...],
    )
    n1, n2, n3 = length.((x1s, x2s, x3s))

    function store_sdf_at_gridpoint(
        i,
        values,
        value_fn,
        sdf_pose,
        sample_origin_pose
    )
        i1, i2, i3 = 1 + i % n1, 1 + (i ÷ n1) % n2, 1 + (i ÷ (n1 * n2)) % n3
        x1, x2, x3 = x1s[i1], x2s[i2], x3s[i3]
        x_local = SA_F32[x1, x2, x3]
        x = sdf_pose * sample_origin_pose * x_local
        values[i1, i2, i3] = value_fn(x)
        return nothing
    end

    n = n1 * n2 * n3
    values = backend(zeros(Float32, n1, n2, n3))
    value_fn = backend(value_fn)
    run_kernel(store_sdf_at_gridpoint, backend, n, values, value_fn, sdf_pose, sample_origin_pose)

    value_itp = Interpolations.interpolate(values, Interpolations.BSpline(Interpolations.Linear()))
    value_itp = Interpolations.extrapolate(value_itp, Interpolations.Line())

    gridded_sdf_pose = sdf_pose * sample_origin_pose
    return GriddedSDF(
        gridded_sdf_pose,
        span,
        real_span,
        resolution,
        values,
        value_itp
    )
end

function GriddedSDF(backend::AbstractComputeBackend, sdf::GeometrySDF; resolution=nothing)
    if resolution === nothing
        lo, hi = sdf.span
        resolution = minimum(hi .- lo) / 20
    end
    return GriddedSDF(backend, sdf.pose, sdf.span, resolution, sdf)
end

function (sdf::GriddedSDF)(x)
    x_local = (inv(sdf.pose) * x .- sdf.real_span[1]) ./ sdf.resolution .+ 1
    return sdf.value_interpolator(x_local...)
end