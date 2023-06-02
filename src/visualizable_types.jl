"Abstract type for visualizable objects."
abstract type Visualizable end

"Default color, based on DodgerBlue4."
const DEFAULT_COLOR = SA_F32[0.06, 0.31, 0.55, 1.00]

export NoVisualizable
struct NoVisualizable <: Visualizable end

export BoxVisualizable
"Visualization object for a box."
Base.@kwdef struct BoxVisualizable{ColorT} <: Visualizable
    pose::Pose6{Float32} = one(Pose6{Float32})
    rgba::SVector{4,Float32} = DEFAULT_COLOR
    color::ColorT = nothing
    span::Span
    wireframe::Bool
end
Adapt.@adapt_structure BoxVisualizable

function BoxVisualizable(box::BoxSDF; rgba::SVector{4,Float32}=DEFAULT_COLOR, color=nothing)
    return BoxVisualizable(; box.pose, rgba, box.span, wireframe=box.dscale <= 0.0, color)
end

export CylinderVisualizable
"Visualization object for a cylinder."
Base.@kwdef struct CylinderVisualizable{ColorT} <: Visualizable
    pose::Pose6{Float32} = one(Pose6{Float32})
    rgba::SVector{4,Float32} = DEFAULT_COLOR
    color::ColorT = nothing
    cap1::SVector{3,Float32}
    cap2::SVector{3,Float32}
    radius::Float32
    wireframe::Bool
end
Adapt.@adapt_structure CylinderVisualizable

function CylinderVisualizable(cyl::CylinderSDF, rgba::SVector{4,Float32}=DEFAULT_COLOR, color=nothing)
    return CylinderVisualizable(; cyl.pose, rgba, cyl.cap1, cyl.cap2, cyl.radius, wireframe=cyl.dscale <= 0.0, color)
end

export MeshVisualizable
"Visualization object for a triangle mesh."
Base.@kwdef struct MeshVisualizable{MeshT,ColorT} <: Visualizable
    pose::Pose6{Float32} = one(Pose6{Float32})
    rgba::SVector{4,Float32} = DEFAULT_COLOR
    color::ColorT = nothing
    mesh::MeshT
end

function MeshVisualizable(filename::String; pose=one(Pose6{Float32}), scale=1.0f0, rgba=DEFAULT_COLOR, color=nothing)
    mesh = load(filename)
    mesh.position .*= scale
    return MeshVisualizable(; pose, rgba, mesh, color)
end

export GenericSDFVisualizable
"Visualization object for a generic SDF, visualizaed with a marching cubes algorithm."
Base.@kwdef struct GenericSDFVisualizable{SDFT,ColorT} <: Visualizable
    pose::Pose6{Float32} = one(Pose6{Float32})
    rgba::SVector{4,Float32} = DEFAULT_COLOR
    color::ColorT = nothing
    sdf::SDFT
end
Adapt.@adapt_structure GenericSDFVisualizable