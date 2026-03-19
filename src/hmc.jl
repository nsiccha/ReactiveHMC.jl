"""
    hmc_state(init; rng, n_steps=1, step_f, stats_f=nothing, min_dham=-1000.)

Create an HMC sampler state with fixed trajectory length.
- `init` — a phasepoint (e.g. from `euclidean_phasepoint`)
- `n_steps` — number of leapfrog steps per iteration
- `step_f` — integrator, e.g. `partial(leapfrog!; stepsize=0.5)`
- `stats_f` — optional trajectory recorder (e.g. `trajectory_stats(dim)`)

After `ReactiveHMC.step!(state)`, the accepted sample is in `state.init.pos`.
"""
@reactive hmc_state(
    init;
    rng,
    n_steps=1,
    min_dham=-1000.,
    step_f=nothing,
    stats_f=nothing
) = begin 
    gofwd = true
    fwd = deepcopy(init)
    dham = 0.
    diverged = !(dham >= min_dham)
    ReactiveHMC.step!(;force=true) = begin 
        init.mom = sqrt(fwd.metric) * randn!(rng, init.mom)
        fwd.pos = init.pos
        fwd.mom = init.mom
        for _ in 1:n_steps
            step_f(fwd)
            dham = finiteorneginf(init.ham - fwd.ham)
            isnothing(stats_f) || stats_f(__self__)
            diverged && return
        end
        randbernoullilog(rng, dham) && rcopy!(init, fwd)
    end
end