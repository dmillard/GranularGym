@precompile_setup begin
    @precompile_all_calls begin
        backend = SinglethreadedCPUBackend()
        nparticles = Int32(1000)
        params = ParticleParams(r=0.001f0)

        fn = joinpath(GranularDynamics.DATA_DIR, "models/helical_gear_short.stl")
        geom = MeshSDF(fn; scale=0.16f0)
        vis = MeshVisualizable(fn; geom.scale)
        geom = GriddedSDF(backend, geom)
        screw = Body(one(Pose6{Float32}), zero(Vel6{Float32}), geom, vis)
        bodies = Bodies(backend.((screw,)))

        pointspan = (SA_F32[0.20, -0.3, 0.2], SA_F32[0.8, 0.3, 2])
        x = uniform_grid_point.(Ref(pointspan), 2 * params.r, 1:nparticles) |> backend
        v = zero(x)

        system = ParticleSystem(;
            x,
            v,
            bodies,
            bodies_prev=deepcopy(bodies),
            params
        ) |> backend

        sim_dt = 5.0f-4
        f_ext = fill(SA_F32[0, 0, -9.8f0], size(x)) |> backend
        solver = ProjectedJacobi(; nsteps=Int32(3), nparticles) |> backend
        stepper = SymplecticEuler(system, sim_dt) |> backend
        step!(backend, system, solver, stepper, f_ext, 0.0f0)
    end
end