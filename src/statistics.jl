"""
    trajectory_stats(dim)

Record all leapfrog positions, gradients, potentials, and Hamiltonian errors
during one NUTS tree expansion. Used as `stats_f` in `nuts_state`.

Call `reset!(tstats, phasepoint)` before each NUTS step.
After the step, access: `positions`, `gradients`, `dhams`, `pots`, `idxs`.

The `idxs` field records tree-building order for animation
(use `invperm(idxs .+ 1) .- 1` for reveal order).
"""
@reactive trajectory_stats(dim) = begin
    positions = ElasticMatrix(zeros(dim, 0))
    gradients = ElasticMatrix(zeros(dim, 0))
    dhams = Float64[]
    pots = Float64[]
    idxs = Int[]
    reset!(phasepoint) = begin
        append!(resize!(positions, dim, 0), phasepoint.pos)
        append!(resize!(gradients, dim, 0), -phasepoint.dham_dpos)
        push!(empty!(dhams), 0)
        push!(empty!(pots), phasepoint.pot)
        push!(empty!(idxs), 0)
    end
    __self__(state) = begin
        f = (state.gofwd ? append! : prepend!)
        f(positions, state.fwd.pos)
        f(gradients, -state.fwd.dham_dpos)
        f(dhams, state.dham)
        f(pots, state.fwd.pot)
        f(idxs, length(idxs))
    end
end

"""
    sampling_stats(tstats)

Accumulate per-iteration statistics across a full NUTS sampling run.
Wraps a `trajectory_stats` instance.

Call `dstats(state, da_state)` after each NUTS step to record:
- `draws` — accepted positions (dim × n_draws)
- `n_steps` — leapfrog steps per iteration
- `stepsizes` — step size used per iteration
- `acc_rate` — acceptance rate per iteration
- `diverged` — divergence flag per iteration
- `full_history` — trajectory positions per iteration (for visualization)
- `full_idxs` — tree-building order per iteration (for animation)
"""
@reactive sampling_stats(tstats) = begin
    dim = tstats.dim
    draws = ElasticMatrix(zeros(dim, 0))
    n_steps = Int[]
    stepsizes = Float64[]
    acc_rate = Float64[]
    diverged = Bool[]
    full_history = Matrix{Float64}[]
    full_idxs = Vector{Int}[]
    __self__(state, da_state) = begin
        append!(draws, state.init.pos)
        push!(n_steps, length(tstats.dhams) - 1)
        push!(stepsizes, state.step_f.stepsize)
        push!(acc_rate, (sum(min1exp, tstats.dhams) - 1) / max(1, length(tstats.dhams) - 1))
        push!(diverged, state.diverged)
        push!(full_history, Matrix(tstats.positions))
        push!(full_idxs, copy(tstats.idxs))
    end
end
