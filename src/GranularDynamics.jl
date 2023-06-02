module GranularDynamics

using SnoopPrecompile
using Requires

import GeometryBasics
import Interpolations
using Adapt
using Base.Threads
using DataStructures
using Dates
using FileIO
using LinearAlgebra
using Primes
using Printf
using Rotations
using SparseArrays
using StaticArrays

DATA_DIR = joinpath(@__DIR__, "../data/")
INIT_FUNCTIONS = Function[]
PRECOMPILE_CALLS = Function[]

include("./backends.jl")
include("./utils.jl")
include("./rigid_body.jl")
include("./lerp.jl")
include("./sdf.jl")
include("./visualizable_types.jl")
include("./particle_system.jl")
include("./solvers.jl")
include("./time_integration.jl")
include("./observation.jl")
include("./actions.jl")
include("./environments.jl")

include("./visualization_stub.jl")
function __init__()
    @require GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a" include("./visualization.jl")
    for init_fn ∈ INIT_FUNCTIONS
        init_fn()
    end
end

include("./precompile.jl")

end