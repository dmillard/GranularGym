export BulldozerEnvironment
struct BulldozerEnvironment{
    BackendT<:AbstractComputeBackend,
    StateT<:AbstractVector{SVector{3,Float32}},
    ImageT<:AbstractArray{Float32,3},
    ScalarT<:AbstractVector{Float32},
    ParticleSystemT<:ParticleSystem,
    RewardSDFT,
    SolverT,
    StepperT,
    VisualizerT,
} <: Environment
    backend::BackendT
    system::ParticleSystemT
    initial_xv::Tuple{StateT,StateT}
    image_ego_cam::ImageT
    image_sky_cam::ImageT
    reward_sdf::RewardSDFT
    reward::ScalarT
    f_ext::StateT
    solver::SolverT
    stepper::StepperT
    visualizer::VisualizerT
    rt_factor::Ref{Float32}
    time::Ref{Float32}
end
Adapt.@adapt_structure BulldozerEnvironment

export BulldozerEnvironment
function BulldozerEnvironment(;
    backend=CUDABackend(),
    nparticles=10000,
    particle_radius=0.02f0,
    particle_mass=1.0f0,
    interparticle_friction=0.5f0,
    energy_loss_factor=0.1f0,
    contact_energy_loss_factor=0.2f0
)
    nparticles = Int32(nparticles)
    params = ParticleParams(
        r=Float32(particle_radius),
        m=Float32(particle_mass),
        μ=Float32(interparticle_friction),
        γ=Float32(energy_loss_factor),
        vd=Float32(contact_energy_loss_factor),
    )

    # Construct rigid bodies
    world_box_sdf = BoxSDF(; span=(SA_F32[-4, -2, 0], SA_F32[4, 2, 4]), dscale=-1.0f0) |> backend
    world_box = Body(; geom=world_box_sdf, viz=BoxVisualizable(world_box_sdf)) |> backend
    bulldozer_fn = joinpath(DATA_DIR, "models/CT_PowerPusher_lowres.stl")
    bulldozer_sdf = MeshSDF(bulldozer_fn) |> backend
    bulldozer_sdf = GriddedSDF(backend, bulldozer_sdf)
    bulldozer_viz = MeshVisualizable(bulldozer_fn; color=:DodgerBlue4)
    bulldozer = Body(; geom=bulldozer_sdf, viz=bulldozer_viz) |> backend
    bodies = Bodies(world_box, bulldozer) |> backend

    goal_box_sdf = BoxSDF(; span=(SA_F32[1, -1, 0], SA_F32[3, 1, 4]), dscale=-1.0f0) |> backend

    # Set up initial state
    point_span = (SA_F32[-2-0.6, -0.6, 0], SA_F32[-2+0.6, 0.6, 4])
    is = 1:nparticles |> collect
    x0 = uniform_grid_point.(Ref(point_span), Ref(params.r * 2.1f0), is) |> backend
    v0 = zero(x0) |> backend
    f_ext = fill(SA_F32[0, 0, -9.8], size(x0)) |> backend
    image_ego_cam = zeros(Float32, 1, 36, 36) |> backend
    image_sky_cam = zeros(Float32, 1, 72, 36) |> backend
    reward = zeros(Float32, 1) |> backend

    sim_dt = 1.0f-3
    system = ParticleSystem(; x=x0, v=v0, bodies, bodies_prev=deepcopy(bodies), params) |> backend
    solver = ProjectedJacobi(; nparticles, nsteps=Int32(3)) |> backend
    stepper = SymplecticEuler(system, sim_dt) |> backend

    # Set initial positions
    #CUDA.@allowscalar begin
    system.bodies.poses[2] = Pose6(bodies.poses[2].R, SA_F32[-3, 0, 0])
    system.bodies_prev.poses[2] = Pose6(bodies.poses[2].R, SA_F32[-3, 0, 0])
    #end

    return BulldozerEnvironment(
        backend,
        system,
        deepcopy.((x0, v0)),
        image_ego_cam,
        image_sky_cam,
        goal_box_sdf,
        reward,
        f_ext,
        solver,
        stepper,
        GranularDynamics.Visualizer[],
        Ref(0.0f0),
        Ref(0.0f0),
    )
end

_pose6_to_xyθdxdy(p::Pose6) = (SA_F32[p.t[1], p.t[2]], atan(p.R[2, 1], p.R[1, 1]), SA_F32[p.R[1, 1], p.R[2, 1]])
_xyθ_to_pose6(xy, θ) = Pose6(RotZ(θ), SA_F32[xy[1], xy[2], 0])

function get_xytheta(env::BulldozerEnvironment)
    #CUDA.@allowscalar begin
    xy, θ, dxdy = _pose6_to_xyθdxdy(env.system.bodies.poses[2])
    return SA_F32[xy[1], xy[2], θ]
    #end
end

get_time_normalized(env::BulldozerEnvironment) = Float32((env.time[] / 30) - 1.0f0)

export render
function render(env::BulldozerEnvironment)
    if isempty(env.visualizer)
        push!(
            env.visualizer,
            GranularDynamics.Visualizer(
                env.backend,
                env.system;
                graphs=String["realtime factor", "reward", "time"],
                images=Dict(
                    "ego cam" => Array(env.image_ego_cam),
                    "sky cam" => Array(env.image_sky_cam),
                )
            )
        )
    end

    reward = #CUDA.@allowscalar begin
        env.reward[1]
    #end
    new_graph_data = Dict(
        "realtime factor" => env.rt_factor[],
        "time" => get_time_normalized(env),
        "reward" => reward,
    )
    new_image_data = Dict(
        "ego cam" => Array(env.image_ego_cam),
        "sky cam" => Array(env.image_sky_cam .* 2 .- 1),
    )
    println(extrema(new_image_data["sky cam"]))
    env.visualizer[1](env.system; new_graph_data, new_image_data)

    return nothing
end

export step!
function step!(env::BulldozerEnvironment, action)
    sim_dt = env.stepper.dt

    # Shift action space so we idle forward a bit
    action += SA_F32[0.1, 0.0]

    nsteps = 100
    duration = @elapsed for _ ∈ 1:nsteps
        #CUDA.@allowscalar begin
        # Handle actions per step
        vl, vθ = action
        current_pose = env.system.bodies.poses[2]
        xy, θ, dxdy = _pose6_to_xyθdxdy(current_pose)
        (x, y) = xy
        dir = vl .* dxdy

        # Respect world boundaries
        (xlo, ylo), (xhi, yhi) = env.system.bodies.geoms[1].span
        x > xhi - 0.5f0 && (dir = SA_F32[min(0, dir[1]), dir[2]])
        x < xlo + 0.5f0 && (dir = SA_F32[max(0, dir[1]), dir[2]])
        y > yhi - 0.5f0 && (dir = SA_F32[dir[1], min(0, dir[2])])
        y < ylo + 0.5f0 && (dir = SA_F32[dir[1], max(0, dir[2])])

        # Set info for velocity calculation kludge
        xy_prev = xy + (sim_dt - 1e-6) * dir
        θ_prev = θ + (sim_dt - 1e-6) * vθ
        xy += sim_dt * dir
        θ += sim_dt * vθ

        env.system.bodies.poses[2] = _xyθ_to_pose6(xy, θ)
        env.system.bodies_prev.poses[2] = _xyθ_to_pose6(xy_prev, θ_prev)
        #end

        step!(env.backend, env.system, env.solver, env.stepper, env.f_ext, env.time[])
    end
    env.rt_factor[] = nsteps * sim_dt / duration
    env.time[] += nsteps * sim_dt

    render_cameras!(env)
    reward!(env)

    #CUDA.synchronize()

    return get_observation(env)
end

export reset!
function reset!(env::BulldozerEnvironment)
    x0, v0 = env.initial_xv
    env.system.x .= x0
    env.system.v .= v0
    env.time[] = 0

    #CUDA.@allowscalar begin
    env.system.bodies.poses[2] = Pose6(RotZ(0.0f0), SA_F32[-3, 0, 0])
    env.system.bodies_prev.poses[2] = Pose6(RotZ(0.0f0), SA_F32[-3, 0, 0])
    env.system.bodies.vels[2] = zero(Vel6{Float32})
    env.system.bodies_prev.vels[2] = zero(Vel6{Float32})
    #end

    render_cameras!(env)

    return get_observation(env)
end


export interactive!
function interactive!(env::BulldozerEnvironment)
    render(env)
    scene = env.visualizer[1].vis.fig.scene
    running = true

    on(scene.events.keyboardbutton) do keyevent
        if keyevent.key == Makie.Keyboard.q && keyevent.action == Makie.Keyboard.release
            running = false
        end
    end

    while running
        keys = scene.events.keyboardstate
        action = SA_F32[0, 0]

        for key ∈ keys
            if key === Makie.Keyboard.up
                action += SA_F32[1.0, 0.0]
            end
            if key === Makie.Keyboard.down
                action += SA_F32[-1.0, 0.0]
            end
            if key === Makie.Keyboard.left
                action += SA_F32[0.0, 1.0]
            end
            if key === Makie.Keyboard.right
                action += SA_F32[0.0, -1.0]
            end
        end

        step!(env, 0.5 * action)
        render(env)
    end
end

function perspective_matrix(fovy, aspect, near, far)
    tanhalf = tan(fovy / 2)
    return SA_F32[
        1/(aspect*tanhalf) 0 0 0
        0 1/tanhalf 0 0
        0 0 (near+far)/(near-far) 2*far*near/(far-near)
        0 0 -1 0
    ]
end

function render_cameras!(env::BulldozerEnvironment)
    fov, near, far = 1.57f0, 0.2f0, 3.0f0
    env.image_ego_cam .= 1
    env.image_sky_cam .= 1
    #CUDA.@allowscalar begin
    current_pose = env.system.bodies.poses[2]
    ego_cam_pose = Pose6(current_pose.R, current_pose.t + SA_F32[0, 0, 0.5])
    world_span = env.system.bodies.geoms[1].span
    x = env.system.x

    # Render particles on GPU
    @show env.backend
    run_kernel(env.backend, length(x), env.image_ego_cam, env.image_sky_cam, x) do i, image_ego_cam, image_sky_cam, x
        xgi = x[i]

        let # Orthogonal sky camera rendering
            lo, hi = world_span
            xc, yc, zc = (xgi .- lo) ./ (hi .- lo)
            c, w, h = size(image_sky_cam)
            xi = Int32(ceil(w * xc))
            yi = Int32(ceil(h * yc))
            if 1 <= xi <= w && 1 <= yi <= h
                depth = 1 - zc
                #CUDA.@atomic
                image_sky_cam[1, xi, yi] = min(image_sky_cam[1, xi, yi], depth)
            end
        end

        let # Perspective ego camera rendering
            xli = inv(ego_cam_pose) * xgi
            xli = SA_F32[xli[2], -xli[3], xli[1]]
            c, w, h = size(image_ego_cam)
            xc, yc, zc, wc = perspective_matrix(fov, Float32(w / h), near, far) * SA_F32[xli..., 1]
            xc, yc, zc = (xc, yc, zc) ./ wc

            if -1 < xc < 1 && -1 < yc < 1 && -1 < zc < 1
                xi = Int32(ceil(w * 0.5f0 * (xc + 1)))
                yi = Int32(ceil(h * 0.5f0 * (yc + 1)))
                if 1 <= xi <= w && 1 <= yi <= h
                    depth = xli[3] / (far + 1)
                    #CUDA.@atomic
                    image_ego_cam[1, xi, yi] = min(image_ego_cam[1, xi, yi], depth)
                end
            end
        end

        return nothing
    end

    return nothing
end

export reward!
function reward!(env)
    env.reward .= 0
    reward_sdf = env.reward_sdf
    factor = Float32(1 / length(env.system.x))
    Base.mapreducedim!(+, env.reward, env.system.x) do xi::SVector{3}
        val = reward_sdf(xi)
        if val > 0
            val = 100
        end
        return factor * val
    end
    return nothing
end

export isdone
function isdone(env)
    return get_time_normalized(env) >= 1
end

function get_observation(env::BulldozerEnvironment)
    return Dict(
        "xytheta" => get_xytheta(env),
        "image_ego_cam" => env.image_ego_cam,
        "image_sky_cam" => env.image_sky_cam,
        "time" => get_time_normalized(env)
    )
end