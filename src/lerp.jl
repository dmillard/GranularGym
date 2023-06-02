function trilerp(vals::AbstractArray, x::AbstractVector)
    lo = Int.(floor.(x))
    hi = Int.(floor.(x)) .+ 1
    xl = x - lo
    xh = hi - x

    if any(lo .< 1) || any(hi .> size(vals))
        # Assumes that we never care about extrapolated gradients
        # and that the boundary of the grid is all positive
        return eltype(vals)(Inf)
    end

    return sum((
        vals[lo[1], lo[2], lo[3]] * xh[1] * xh[2] * xh[3],
        vals[hi[1], lo[2], lo[3]] * xl[1] * xh[2] * xh[3],
        vals[lo[1], hi[2], lo[3]] * xh[1] * xl[2] * xh[3],
        vals[hi[1], hi[2], lo[3]] * xl[1] * xl[2] * xh[3],
        vals[lo[1], lo[2], hi[3]] * xh[1] * xh[2] * xl[3],
        vals[hi[1], lo[2], hi[3]] * xl[1] * xh[2] * xl[3],
        vals[lo[1], hi[2], hi[3]] * xh[1] * xl[2] * xl[3],
        vals[hi[1], hi[2], hi[3]] * xl[1] * xl[2] * xl[3],
    ))
end