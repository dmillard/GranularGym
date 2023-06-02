# This file is conditionally loaded with Requires.jl if GLMakie is loaded.

using .GLMakie
import ColorSchemes
import MarchingCubes
import Meshes
import MeshViz

function make_scene_object!(ax, sdf_viz::GenericSDFVisualizable)
    if sdf_viz.sdf isa GriddedSDF
        resolution = sdf_viz.sdf.resolution
        gridded_sdf = sdf_viz.sdf
    else
        lo, hi = sdf_viz.sdf.span
        resolution = minimum(hi .- lo) / 20
        gridded_sdf = GriddedSDF(sdf_viz.sdf; resolution)
    end

    mc = MarchingCubes.MC(Array(gridded_sdf.values), Int32)
    MarchingCubes.march(mc)

    for i ∈ eachindex(mc.vertices)
        mc.vertices[i] *= resolution
        mc.vertices[i] += gridded_sdf.real_span[1]
    end

    color = if sdf_viz.color !== nothing
        sdf_viz.color
    else
        Makie.RGBAf(sdf_viz.rgba...)
    end

    mesh = MarchingCubes.makemesh(Meshes, mc)
    scenemesh = MeshViz.viz!(ax, mesh; color)

    return scenemesh
end

function make_scene_object!(ax, box_viz::BoxVisualizable)
    lo, hi = map(v -> Makie.GeometryBasics.Vec3(v...), box_viz.span)
    mesh = Makie.GeometryBasics.HyperRectangle(lo, (hi - lo))

    color = if box_viz.color !== nothing
        box_viz.color
    else
        Makie.RGBAf(box_viz.rgba...)
    end

    return if box_viz.wireframe
        wireframe!(ax, mesh)
    else
        mesh!(ax, mesh, color=color)
    end
end

function make_scene_object!(ax, cyl_viz::CylinderVisualizable)
    cap1, cap2 = map(
        v -> Makie.GeometryBasics.Vec3(v...),
        (cyl_viz.cap1, cyl_viz.cap2)
    )
    mesh = Makie.GeometryBasics.Cylinder3{Float32}(cap1, cap2, cyl_viz.radius)

    color = if cyl_viz.color !== nothing
        cyl_viz.color
    else
        Makie.RGBAf(cyl_viz.rgba...)
    end

    return if cyl_viz.wireframe
        wireframe!(ax, mesh)
    else
        mesh!(ax, mesh, color=color)
    end
end

function make_scene_object!(ax, mesh_viz::MeshVisualizable)
    color = if mesh_viz.color !== nothing
        mesh_viz.color
    else
        Makie.RGBAf(mesh_viz.rgba...)
    end
    return mesh!(ax, mesh_viz.mesh, color=color)
end

function visualize!(ax, pose_obs::Observable{<:Pose6}, viz::Visualizable)
    if viz isa NoVisualizable
        return nothing
    end
    scenemesh = make_scene_object!(ax, viz)

    on(pose_obs) do pose
        pose_concat = pose * viz.pose
        q = QuatRotation(pose_concat.R)
        Makie.rotate!(scenemesh, Quaternion(q.x, q.y, q.z, q.w))
        Makie.translate!(scenemesh, pose_concat.t...)
    end

    return nothing
end

struct Visualizer_{T,VT<:AbstractVector{SVector{3,T}}}
    fig::Figure
    sim_ax::Axis3
    points::Observable{VT}
    vels::Observable{VT}
    body_poses::Vector{Observable{Pose6{T}}}
    graphs::Dict{String,Tuple{Axis,Observable{CircularBuffer{Float32}}}}
    images::Dict{String,Tuple{Axis,Observable{Array{Float32,3}}}}
end

function Visualizer_(
    backend::AbstractComputeBackend,
    system::ParticleSystem;
    title="",
    colorby=:height,
    graphs=Vector{String}(),
    images=Dict()
)
    resolution = (1200, 1200)
    screen = GLMakie.singleton_screen(false)
    nparticles = length(system.x)

    points = Observable(system.x)
    vels = Observable(system.v)
    body_poses = Vector(system.bodies.poses)

    scene = Scene(; resolution)
    fig = Figure(; scene)
    sim_ax = Axis3(
        fig[1, 1];
        title,
        titlegap=0,
        aspect=:data,
        viewmode=:fit,
        halign=:center,
        valign=:top,
        width=Fixed(1200),
        height=Fixed(1000)
    )

    points3f = if backend isa CUDABackend
        cuda_observable_map(Point3f, nparticles, points) do array, points
            i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
            if i <= nparticles
                @inbounds array[i] = Point3f(points[i]...)
            end
            return nothing
        end
    else
        map(points) do points
            map(Array(points)) do point
                Point3f(point...)
            end
        end
    end

    velnorms = if backend isa CUDABackend
        out = Observable(2 .* CUDA.rand(nparticles))

        on(vels) do vels
            cuda_run_kernel(nparticles, out[], vels) do out, vels
                i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
                if i <= nparticles
                    @inbounds out[i] = norm(vels[i])
                end
                return nothing
            end
            notify(out)
        end

        out
    else
        map(vels -> norm.(eachcol(vels)), vels)
    end

    if colorby == :height
        color = map(points) do points
            map(p -> p[3], points)
        end
        colorrange = map(system.bodies.geoms[1].span) do v
            v[3]
        end
        colorrange = (-0.5, 0.7)
    elseif colorby == :velocity
        color = velnorms
        colorrange = (0, 10)
    end

    scatter!(
        sim_ax,
        points3f;
        color,
        colorrange,
        colormap=Reverse(ColorSchemes.turbid),
        markersize=2 * system.params.r,
        markerspace=:data
    )

    body_poses_obs = Observable{eltype(body_poses)}[]
    body_poses_obs = Observable.(body_poses)
    visualize!.(Ref(sim_ax), body_poses_obs, system.bodies.vizs)

    lo, hi = system.bodies.geoms[1].span
    lo = lo .- 0.1
    hi = hi .+ 0.1
    limits!(sim_ax, lo[1], hi[1], lo[2], hi[2], lo[3], hi[3])

    grid_rows = 2
    grid_cols = maximum(length.((graphs, images)))
    display_grid = GridLayout(
        fig[2, 1],
        grid_rows,
        grid_cols;
        valign=:bottom,
        halign=:left,
        rowsizes=fill(Fixed(100), grid_rows),
        colsizes=fill(Fixed(100), grid_cols)
    )

    stored_graphs = Dict{String,Tuple{Axis,Observable{CircularBuffer{Float32}}}}()
    for (i, title) ∈ enumerate(graphs)
        graph_data = Observable(CircularBuffer{Float32}(100))
        append!(graph_data[], zeros(capacity(graph_data[])))

        graph_ax = Axis(
            display_grid[1, i],
            title=title,
            ytickformat="{:0.2f}",
            yaxisposition=:right,
        )
        hidexdecorations!(graph_ax)
        lines!(
            graph_ax,
            graph_data;
            color=graph_data,
            colorrange=(0.8, 1.2),
            colormap=ColorSchemes.RdYlGn_10
        )
        push!(stored_graphs, title => (graph_ax, graph_data))
    end

    stored_images = Dict{String,Tuple{Axis,Observable{Array{Float32,3}}}}()
    for (i, (title, im0)) ∈ enumerate(images)
        image_data = Observable(im0)

        image_ax = Axis(
            display_grid[2, i],
            title=title,
            aspect=DataAspect(),
        )
        hidedecorations!(image_ax)
        image!(
            image_ax,
            map(im -> dropdims(im; dims=1), image_data);
            interpolate=false,
            colorrange=(0.01, 0.99),
            lowclip=:black,
            highclip=:black,
            colormap=[:white, :black]
        )
        push!(stored_images, title => (image_ax, image_data))
    end

    set_window_config!(focus_on_show=true, title="GranularDynamics.jl")
    resize_to_layout!(fig)
    display(fig)

    return Visualizer_(fig, sim_ax, points, vels, body_poses_obs, stored_graphs, stored_images)
end

function (vis::Visualizer_)(system; new_graph_data=Dict(), new_image_data=Dict())
    body_poses = Array(system.bodies.poses)

    vis.points[] = system.x
    vis.vels[] = system.v
    for (i, pose) ∈ enumerate(body_poses)
        vis.body_poses[i][] = pose
    end
    for (name, graph_data) in new_graph_data
        ax, obs = vis.graphs[name]
        push!(obs[], graph_data)
        notify(obs)
        lo, hi = extrema(obs[])
        ylims!(ax, lo - 0.1, hi + 0.1)
        ax.yticks = WilkinsonTicks(3)
    end
    for (name, image_data) in new_image_data
        ax, obs = vis.images[name]
        obs[] .= image_data
        notify(obs)
    end
end

function cuda_observable_map(f, result_type::Type, size, cuda_obs::Observable)
    buffer = Observable(GLMakie.GLAbstraction.GLBuffer(result_type, size))
    resource = let
        ref = Ref{CUDA.CUgraphicsResource}()
        CUDA.cuGraphicsGLRegisterBuffer(
            ref,
            buffer[].id,
            CUDA.CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD
        )
        ref[]
    end
    on(cuda_obs) do cuda_obs
        # Reference: https://discourse.julialang.org/t/cuarray-glmakie/52461/8
        # map OpenGL buffer object for writing from CUDA
        CUDA.cuGraphicsMapResources(1, [resource], CUDA.stream())

        # get a CuArray object that we can work with
        array = let
            ptr_ref = Ref{CUDA.CUdeviceptr}()
            numbytes_ref = Ref{Csize_t}()
            CUDA.cuGraphicsResourceGetMappedPointer_v2(
                ptr_ref,
                numbytes_ref,
                resource
            )

            ptr = reinterpret(CuPtr{result_type}, ptr_ref[])
            len = Int(numbytes_ref[] ÷ sizeof(result_type))

            unsafe_wrap(CuArray, ptr, len)
        end

        # generate points
        cuda_run_kernel(f, size, array, cuda_obs)

        synchronize() # wait for the GPU to finish
        CUDA.cuGraphicsUnmapResources(1, [resource], CUDA.stream())
    end

    buffer
end