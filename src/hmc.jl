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
            stats_f(__self__)
            diverged && return
        end
        randbernoullilog(rng, dham) && rcopy!(init, fwd)
    end
end