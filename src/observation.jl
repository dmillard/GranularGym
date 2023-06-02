abstract type Observation end

struct AxisAlignedHeightField{T} <: Observation
    xrange::SVector{2,T}
    yrange::SVector{2,T}
    cellsize::T
end

function y_zero(obs::AxisAlignedHeightField, zeros_f=zeros)
    xcells = Int(ceil(diff(obs.xrange)[1] / obs.cellsize))
    ycells = Int(ceil(diff(obs.yrange)[1] / obs.cellsize))
    return zeros_f(xcells, ycells)
end

function observe!(y::AbstractMatrix, obs::AxisAlignedHeightField, system::ParticleSystem)
    y .= -Inf
    nparticles = size(system.x, 2)
    rcell = Int(ceil(system.params.r / obs.cellsize))
    rsq = system.params.r^2
    for i ∈ 1:nparticles
        xi = system.x[i]
        if !all(isfinite.(xi))
            println("NonFinites detected! Count: $(sum(isnan.(system.x))), Idx: $i")
            continue
        end
        xcentercell = Int(ceil((xi[1] - obs.xrange[1]) / obs.cellsize))
        ycentercell = Int(ceil((xi[2] - obs.yrange[1]) / obs.cellsize))
        for dx ∈ -rcell:rcell, dy ∈ -rcell:rcell
            xcell = xcentercell + dx
            ycell = ycentercell + dy
            if !all((0, 0) .< (xcell, ycell) .<= size(y))
                continue
            end
            if dx * dx + dy * dy > rcell * rcell
                continue
            end
            if xi[3] > y[xcell, ycell]
                dsq = (dx * dx + dy * dy) * obs.cellsize^2
                z = sqrt(max(0, rsq - dsq))
                y[xcell, ycell] = xi[3] + z
            end
        end
    end

    for I ∈ eachindex(y)
        if y[I] == -Inf
            y[I] = 0
        end
    end

    return y
end