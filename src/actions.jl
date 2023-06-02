export ToolWaypoint
struct ToolWaypoint{T}
    pose::Pose6{T}
    time::T
end

export ToolTrajectory
struct ToolTrajectory{T,VWT<:AbstractVector{ToolWaypoint{T}}}
    waypoints::VWT
end

function interpolate_position(traj::ToolTrajectory, time)
    i = 1
    while traj.waypoints[i].time < time
        i += 1
    end

    prev, next = traj.waypoints[i-1], traj.waypoints[i]
    qprev, qnext = QuatRotation(prev.pose.R), QuatRotation(next.pose.R)
    frac = (time - prev.time) / (next.time - prev.time)

    R = slerp(qprev, qnext, frac)
    t = (next.pose.t - prev.pose.t) * frac + prev.pose.t

    return Pose6(SMatrix(R), t)
end

function log_rodrigues(m::AbstractMatrix)
    if m == I
        return SA_F32[0, 0, 0]
    elseif tr(m) == -1
        ωhat = SA_F32[m[1, 3], m[2, 3], 1+m[3, 3]] / sqrt(2 * (1 + m[3, 3]))
        return π * ωhat
    else
        θ = acos((tr(m) - 1) / 2)
        ωcross = (m - m') / (2 * sin(θ))
        ωhat = SA_F32[-ωcross[1, 2], ωcross[1, 3], -ωcross[2, 3]]
        return θ * ωhat
    end
end

function interpolate_velocity(traj::ToolTrajectory, time)
    dt = 1e-3
    t1, t2 = if time <= 2 * dt
        dt, 2 * dt
    else
        time - dt, time
    end

    p1 = interpolate_position(traj, t1)
    p2 = interpolate_position(traj, t2)
    v = (p2.t - p1.t) ./ dt
    ω = log_rodrigues(p2.R \ p1.R) ./ dt

    return Vel6(ω, v)
end

export RobotWaypoint
struct RobotWaypoint{T,QT<:AbstractVector{T}}
    q::QT
    time::T
end
Adapt.@adapt_structure RobotWaypoint

export RobotTrajectory
struct RobotTrajectory{VWT<:AbstractVector{<:RobotWaypoint}}
    waypoints::VWT
end
Adapt.@adapt_structure RobotTrajectory

export interpolate_configuration!
function interpolate_configuration!(q::AbstractVector, traj::RobotTrajectory, time)
    ti = 1
    while traj.waypoints[ti].time < time
        ti += 1
    end

    prev = traj.waypoints[ti-1]
    next = traj.waypoints[ti]
    frac = (time - prev.time) / (next.time - prev.time)

    for qi ∈ eachindex(q)
        q[qi] = (next.q[qi] - prev.q[qi]) * frac + prev.q[qi]
    end

    return nothing
end