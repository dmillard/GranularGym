Base.@kwdef struct YPPPYRobot
    base::Pose6{Float32}
    lengths::NTuple{2,Float32}
    vel_limits::NTuple{5,Float32}
end
Adapt.@adapt_structure YPPPYRobot

export home_configuration
function home_configuration(::Union{Type{YPPPYRobot},YPPPYRobot})
    #SA_F32[0.0, -π/3, 2π/3, -π/4, 0]
    SA_F32[0.0, -π/2, π, 0, 0]
end

export YPPPYTestbedEnvironment
Base.@kwdef struct YPPPYTestbedEnvironment{EnvBodiesT,ParamsT<:ParticleParams,AT} <: Environment
    env_bodies::EnvBodiesT
    params::ParamsT
    solver_steps::Int32
    robot::YPPPYRobot
    initial_conditions::Tuple{AT,AT}
    nparticles::Int32
    sim_dt::Float32
    hash_table_size::Int32
end
Adapt.@adapt_structure YPPPYTestbedEnvironment

function world_box(::Type{YPPPYTestbedEnvironment})
    world_box_sdf = BoxSDF(
        one(Pose6{Float32}),
        (SA_F32[-2, -2, -2], SA_F32[2, 2, 2]),
        -1.0f0
    )
    return Body(
        Pose6(SMatrix(RotX{Float32}(0.0)), SA_F32[0, 0, 2]),
        Vel6(SA_F32[0, 0, 0], SA_F32[0, 0, 0]),
        world_box_sdf,
        BoxVisualizable(world_box_sdf)
    )
end

function outer_box(::Type{YPPPYTestbedEnvironment}, xspan=1.5, yspan=1.5, zspan=0.25, thickness=0.10)
    x, y, z, t = 2 * xspan, 2 * yspan, 2 * zspan, 2 * thickness
    geom = CompositeGeometrySDF(
        Pose6(SMatrix(RotY{Float32}(0.0)), SA_F32[0, 0, 0]),
        (
            BoxSDF(Pose6(SMatrix(RotY{Float32}(0.0)), SA_F32[-x/2+t/2, 0, 0]), centered_span(SA_F32[t, y, z]), 1.0f0),
            BoxSDF(Pose6(SMatrix(RotY{Float32}(0.0)), SA_F32[x/2-t/2, 0, 0]), centered_span(SA_F32[t, y, z]), 1.0f0),
            BoxSDF(Pose6(SMatrix(RotY{Float32}(0.0)), SA_F32[0, 0, -z/2+t/2]), centered_span(SA_F32[x, y, t]), 1.0f0),
            BoxSDF(Pose6(SMatrix(RotY{Float32}(0.0)), SA_F32[0, -y/2+t/2, 0]), centered_span(SA_F32[x, t, z]), 1.0f0),
            BoxSDF(Pose6(SMatrix(RotY{Float32}(0.0)), SA_F32[0, y/2-t/2, 0]), centered_span(SA_F32[x, t, z]), 1.0f0),
        ),
    )
    return Body(
        Pose6(SMatrix(RotX{Float32}(0.0)), SA_F32[-0.5, 0.0, 0.25]),
        Vel6(SA_F32[0, 0, 0], SA_F32[0.0, 0.0, 0.0]),
        GriddedSDF(geom),
        GenericSDFVisualizable(; geom.pose, sdf=geom),
    )
end

function scoop(::Type{YPPPYTestbedEnvironment}, xspan=0.7, yspan=0.7, zspan=0.3, thickness=0.10)
    x, y, z, t = xspan, yspan, zspan, thickness
    geom = CompositeGeometrySDF(
        Pose6(SMatrix(RotY{Float32}(0.0)), SA_F32[x/2, 0, -z/2]),
        (
            BoxSDF(Pose6(SMatrix(RotY{Float32}(0.0)), SA_F32[-x/2+t/2, 0, 0]), centered_span(SA_F32[t, y, z]), 1.0f0),
            BoxSDF(Pose6(SMatrix(RotY{Float32}(0.0)), SA_F32[0, 0, -z/2+t/2]), centered_span(SA_F32[x, y, t]), 1.0f0),
            BoxSDF(Pose6(SMatrix(RotY{Float32}(0.0)), SA_F32[0, -y/2+t/2, 0]), centered_span(SA_F32[x, t, z]), 1.0f0),
            BoxSDF(Pose6(SMatrix(RotY{Float32}(0.0)), SA_F32[0, y/2-t/2, 0]), centered_span(SA_F32[x, t, z]), 1.0f0),
        ),
    )
    return Body(
        Pose6(SMatrix(RotX{Float32}(0.0)), SA_F32[0.0, 0.0, 0.0]),
        Vel6(SA_F32[0, 0, 0], SA_F32[0.0, 0.0, 0.0]),
        geom,
        GenericSDFVisualizable(; geom.pose, sdf=geom),
    )
end

function create_bodies(robot::YPPPYRobot, tool::Body)
    t = 0.2
    base_geom = BoxSDF(
        Pose6(SMatrix(RotY{Float32}(0.0)), SA_F32[0, 0, 0]),
        centered_span(SA_F32[2t, 2t, 2t]),
        1.0f0
    )
    link1_geom = BoxSDF(
        Pose6(SMatrix(RotY{Float32}(0.0)), SA_F32[0, 0, 0]),
        (SA_F32[0, -t/2, -t/2], SA_F32[robot.lengths[1], t/2, t/2]),
        1.0f0
    )
    link2_geom = BoxSDF(
        Pose6(SMatrix(RotY{Float32}(0.0)), SA_F32[0, 0, 0]),
        (SA_F32[0, -t/2, -t/2], SA_F32[robot.lengths[2], t/2, t/2]),
        1.0f0
    )
    return (
        # Base
        Body(
            one(Pose6{Float32}),
            zero(Vel6{Float32}),
            base_geom,
            BoxVisualizable(base_geom),
        ),
        # Arm Link 1
        Body(
            one(Pose6{Float32}),
            zero(Vel6{Float32}),
            link1_geom,
            BoxVisualizable(link1_geom),
        ),
        # Arm Link 2
        Body(
            one(Pose6{Float32}),
            zero(Vel6{Float32}),
            link2_geom,
            BoxVisualizable(link2_geom),
        ),
        tool
    )
end

function fk!(poses::AbstractVector, ids, robot::YPPPYRobot, q::AbstractVector)
    poses[ids[1]] = robot.base * Pose6(SMatrix(RotZ{Float32}(q[1])), SA_F32[0, 0, 0])
    poses[ids[2]] = poses[ids[1]] * Pose6(SMatrix(RotY{Float32}(q[2])), SA_F32[0, 0, 0])
    poses[ids[3]] = poses[ids[2]] * Pose6(SMatrix(RotY{Float32}(q[3])), SA_F32[robot.lengths[1], 0, 0])
    # Link 4 doesn't actually have a body associated with it
    poses[ids[4]] = poses[ids[3]] *
                    Pose6(SMatrix(RotY{Float32}(q[4])), SA_F32[robot.lengths[2], 0, 0]) *
                    Pose6(SMatrix(RotZ{Float32}(q[5])), SA_F32[0, 0, 0])

    return nothing
end

function initial_x_v(env_bodies::Tuple; nparticles::Int32, solver_steps::Int32, params::ParticleParams, sim_dt::Real, hash_table_size::Int32, T::Real=3.0)
    x = zeros(SVector{3,Float32}, nparticles)
    xi = zeros(3)
    for i ∈ 1:length(x)
        s = Int(round(exp(log(nparticles) / 3)))
        xi .= 0
        xi[1] = i % s
        xi[2] = (i ÷ s) % s
        xi[3] = (i ÷ s^2) % s
        xi ./= s
        xi .-= 0.5 - 1 / (2 * s)
        xi .*= [1.8, 1.8, 1]
        xi .+= SA_F32[-0.5, 0, 1]
        xi .+= randn(3) / 100
        x[i] = xi
    end

    v = zero(x)

    g = zero(x)
    g .= Ref(SA_F32[0, 0, -9.8f0])
    bodies = Bodies(env_bodies...)
    bodies_prev = deepcopy(bodies)

    system = ParticleSystem(; x, v, bodies, bodies_prev, params)
    solver = ProjectedJacobi(; nsteps=solver_steps, nparticles, hash_table_size)
    stepper = SymplecticEuler(system, sim_dt)

    nsteps = 0# Int(round(T / sim_dt))
    for step ∈ 1:nsteps
        t = step * sim_dt
        step!(system, stepper, g, t; solver)
    end

    return system.x, system.v
end

function YPPPYTestbedEnvironment(
    nparticles::Int32;
    solver_steps::Int32=Int32(10),
    params=ParticleParams(),
    sim_dt=1.0f0 / 1000,
    hash_table_size=Int32(2) * nparticles,
    gpu=false
)
    robot_base = Pose6(SMatrix(RotX{Float32}(0.0)), SA_F32[-3, 0, 1])
    robot = YPPPYRobot(;
        base=robot_base,
        lengths=(2.0, 2.0),
        vel_limits=Tuple(1e-1 for i in 1:5)
    )

    env_bodies = (world_box(YPPPYTestbedEnvironment), outer_box(YPPPYTestbedEnvironment),)
    x0, v0 = if gpu
        cuda_initial_x_v(env_bodies; solver_steps, params, nparticles, sim_dt, hash_table_size)
    else
        initial_x_v(env_bodies; solver_steps, params, nparticles, sim_dt, hash_table_size)
    end

    return YPPPYTestbedEnvironment(;
        nparticles,
        params,
        solver_steps,
        sim_dt,
        env_bodies,
        robot,
        initial_conditions=(x0, v0),
        hash_table_size
    )
end

function make_trajectory(
    environment::YPPPYTestbedEnvironment,
    action::AbstractVector;
    start=home_configuration(environment.robot)
)
    @views x = reshape([start; action], 5, :)
    dists = diff(x; dims=2)
    times = dists ./ environment.robot.vel_limits
    dt = maximum(times; dims=1)
    ts = [[0]; cumsum(dt'; dims=1)]

    return RobotTrajectory(map(zip(ts, eachcol(x))) do (t, xi)
        xi = SA_F32[xi[1], xi[2], xi[3], xi[4], xi[5]]
        return RobotWaypoint(xi, Float32(t))
    end)
end

export sample_action
function sample_action(nwaypoints::Integer; environment::YPPPYTestbedEnvironment)
    n = nwaypoints
    return vec(randn(5, n) / 3 .+ home_configuration(environment.robot))
end

export get_action_lb_ub
function get_action_lb_ub(nwaypoints::Integer; environment::YPPPYTestbedEnvironment)
    lb = home_configuration(environment.robot) .- π / 6
    ub = home_configuration(environment.robot) .+ π / 6
    return vec(repeat(lb, outer=(1, nwaypoints))), vec(repeat(ub, outer=(1, nwaypoints)))
end

export evaluate_action
function evaluate_action(
    action::AbstractVector;
    environment::YPPPYTestbedEnvironment,
    visualize::Bool=false,
    record::Bool=false,
    log_callback::Union{Function,Nothing}=nothing
)
    x0, v0 = deepcopy(environment.initial_conditions)
    traj = make_trajectory(environment, action)
    T = traj.waypoints[end].time - environment.sim_dt

    g = zero(x0)
    g .= Ref(SA_F32[0, 0, -9.8])

    bodies = Bodies(environment.env_bodies..., create_bodies(environment.robot, scoop(YPPPYTestbedEnvironment))...)
    bodies_prev = deepcopy(bodies)
    robot_body_ids = (length(environment.env_bodies)+1):length(bodies.geoms)

    obs = AxisAlignedHeightField(SA_F32[-2, 0], SA_F32[-1, 1], 0.1f0)
    y = y_zero(obs)

    system = ParticleSystem(; x=x0, v=v0, bodies, bodies_prev, environment.params)
    solver = ProjectedJacobi(;
        nsteps=environment.solver_steps,
        nparticles=environment.nparticles,
        hash_table_size=environment.hash_table_size
    )
    stepper = SymplecticEuler(system, environment.sim_dt)

    fps = 60
    if visualize
        clean_typeof(::T) where {T} = (isempty(T.parameters) ? T : T.name.wrapper)
        title = "$(clean_typeof(solver)), $(clean_typeof(stepper)), n=$(size(x0, 2)), dt=$(environment.sim_dt)"
        vis = Visualizer(
            system;
            title,
            graphs=String["realtime factor"]
        )

        if record
            videostream = GLMakie.VideoStream(vis.fig.scene; framerate=fps)
        end
    end

    q = Array(home_configuration(environment.robot))
    q_prev = Array(home_configuration(environment.robot))

    last_frame = 0
    nsteps = Int(round(T / environment.sim_dt))
    compute_time = @elapsed for step ∈ 1:nsteps
        t = step * environment.sim_dt

        interpolate_configuration!(q, traj, t)
        interpolate_configuration!(q_prev, traj, max(0, t - 1e-6))

        fk!(
            system.bodies.poses,
            robot_body_ids,
            environment.robot,
            q,
        )
        fk!(
            system.bodies_prev.poses,
            robot_body_ids,
            environment.robot,
            q_prev,
        )

        duration = @elapsed begin
            step!(system, stepper, g, t; solver)
        end

        if visualize && step * environment.sim_dt - last_frame >= 1 / fps
            vis(
                system;
                new_graph_data=Dict("realtime factor" => environment.sim_dt / duration)
            )
            sleep(max(0, environment.sim_dt - duration))
            last_frame = step * environment.sim_dt

            if record
                GLMakie.recordframe!(videostream)
            end
        end
    end


    if record
        GLMakie.save("moving_plate_$(now()).mp4", videostream)
    end

    cost = sum(y)
    if log_callback !== nothing
        waypoint_pairs = map(enumerate(eachcol(reshape(action, 5, :)))) do (i, waypoint)
            Symbol.((
                "q1_$i",
                "q2_$i",
                "q3_$i",
                "q4_$i",
                "q5_$i",
            )) .=> waypoint
        end
        waypoint_pairs = reduce(vcat, waypoint_pairs)
        log_callback([:cost => cost, waypoint_pairs...])
    end

    @printf "%4.3f RT factor: %0.2f (%0.2f seconds)\n" cost (environment.sim_dt * nsteps / compute_time) compute_time

    return cost
end