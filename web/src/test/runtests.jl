using TestModules
using Random, ReactiveHMC, ReactiveObjects, LinearAlgebra, Statistics, ElasticArrays

# --- Shared setup ---

DIM() = 2

pot_f(x) = 0.5 * dot(x, x)
function grad_f(x)
    (pot_f(x), copy(x))
end

function make_phasepoint(pos, mom; metric=Diagonal(ones(DIM())))
    euclidean_phasepoint(pot_f, grad_f, metric, pos, mom)
end

struct _StepFn; stepsize::Float64; end
(s::_StepFn)(pp) = leapfrog!(pp; stepsize=s.stepsize)

# --- Tests ---

@testset "Euclidean phasepoint fields" begin
    pos = [1.0, 2.0]
    mom = [0.5, -0.3]
    pp = make_phasepoint(pos, mom)
    @test pp.pot ≈ 2.5
    expected_kin = 0.5 * dot(mom, mom)
    @test pp.ham ≈ pp.pot + expected_kin
    @test pp.dham_dpos ≈ pos
    @test pp.dham_dmom ≈ mom
end

@testset "Leapfrog approximate Hamiltonian preservation" begin
    pos = [1.0, 2.0]
    mom = [0.5, -0.3]
    pp = make_phasepoint(pos, mom)
    ham_before = pp.ham
    leapfrog!(pp; stepsize=0.01)
    @test pp.ham ≈ ham_before atol=1e-4
end

@testset "Leapfrog reversibility" begin
    pos = [1.0, 2.0]
    mom = [0.5, -0.3]
    pp = make_phasepoint(copy(pos), copy(mom))
    leapfrog!(pp; stepsize=0.1)
    @invalidatedependants! pp.mom = -pp.mom
    leapfrog!(pp; stepsize=0.1)
    @invalidatedependants! pp.mom = -pp.mom
    @test pp.pos ≈ pos atol=1e-12
    @test pp.mom ≈ mom atol=1e-12
end

@testset "Multistep equivalence" begin
    pos = [1.0, 2.0]
    mom = [0.5, -0.3]
    stepsize = 0.2
    pp1 = make_phasepoint(copy(pos), copy(mom))
    for _ in 1:4
        leapfrog!(pp1; stepsize=stepsize/4)
    end
    pp2 = make_phasepoint(copy(pos), copy(mom))
    multistep(leapfrog!, pp2; n_steps=4, stepsize=stepsize)
    @test pp1.pos ≈ pp2.pos atol=1e-12
    @test pp1.mom ≈ pp2.mom atol=1e-12
    pp3 = make_phasepoint(copy(pos), copy(mom))
    step_fn = multistep(leapfrog!; n_steps=4)
    step_fn(pp3; stepsize=stepsize)
    @test pp1.pos ≈ pp3.pos atol=1e-12
end

@testset "Welford variance" begin
    rng = Xoshiro(123)
    wv = welford_var(DIM())
    n = 5000
    data = [randn(rng, DIM()) for _ in 1:n]
    for x in data
        step!(wv, x)
    end
    expected_mean = mean(data)
    expected_var = var(data; corrected=false)
    @test wv.n ≈ n
    @test wv.mean ≈ expected_mean atol=0.05
    @test wv.var ≈ expected_var atol=0.05
end

@testset "Welford variance with matrix input" begin
    rng = Xoshiro(456)
    wv = welford_var(DIM())
    data = randn(rng, DIM(), 100)
    step!(wv, data)
    @test wv.n ≈ 100
    @test wv.mean ≈ vec(mean(data; dims=2)) atol=0.2
end

@testset "Dual averaging convergence" begin
    da = dual_averaging_state(1.0)
    for _ in 1:200
        fit!(da, 0.95)
    end
    stepsize_high = da.current
    da2 = dual_averaging_state(1.0)
    for _ in 1:200
        fit!(da2, 0.3)
    end
    stepsize_low = da2.current
    @test stepsize_high > stepsize_low
    @test isfinite(da.current)
    @test isfinite(da.final)
    @test da.current > 0
    @test da2.current > 0
end

@testset "NUTS sampler basic" begin
    rng = Xoshiro(42)
    pos0 = zeros(DIM())
    mom0 = randn(rng, DIM())
    pp = make_phasepoint(copy(pos0), copy(mom0))
    state = nuts_state(pp; rng=rng, step_f=phasepoint -> leapfrog!(phasepoint; stepsize=0.5))
    for _ in 1:20
        @invalidatedependants! state.init.mom = randn(rng, DIM())
        step!(state)
    end
    @test all(isfinite, state.init.pos)
    @test norm(state.init.pos) < 20.0
end

@testset "Trajectory stats recording" begin
    rng = Xoshiro(99)
    pp = make_phasepoint(zeros(DIM()), randn(rng, DIM()))
    tstats = trajectory_stats(DIM())
    state = nuts_state(pp;
        rng=rng,
        step_f=phasepoint -> leapfrog!(phasepoint; stepsize=0.3),
        stats_f=tstats
    )
    reset!(tstats, state.init)
    @invalidatedependants! state.init.mom = randn(rng, DIM())
    step!(state)
    @test size(tstats.positions, 1) == DIM()
    @test size(tstats.positions, 2) >= 1
    @test length(tstats.dhams) >= 1
    @test length(tstats.pots) >= 1
    @test length(tstats.idxs) >= 1
end

@testset "Sampling stats accumulation" begin
    rng = Xoshiro(77)
    pp = make_phasepoint(zeros(DIM()), randn(rng, DIM()))
    tstats = trajectory_stats(DIM())
    dstats = sampling_stats(tstats)
    da = dual_averaging_state(0.5)
    state = nuts_state(pp;
        rng=rng,
        step_f=_StepFn(0.5),
        stats_f=tstats
    )
    n_iter = 5
    for _ in 1:n_iter
        reset!(tstats, state.init)
        @invalidatedependants! state.init.mom = randn(rng, DIM())
        step!(state)
        dstats(state, da)
    end
    @test size(dstats.draws, 2) == n_iter
    @test length(dstats.n_steps) == n_iter
    @test length(dstats.stepsizes) == n_iter
    @test length(dstats.acc_rate) == n_iter
    @test length(dstats.diverged) == n_iter
    @test all(isfinite, dstats.draws)
end

@testset "Trajectory stats reset!" begin
    rng = Xoshiro(55)
    pp = make_phasepoint([1.0, 2.0], randn(rng, DIM()))
    tstats = trajectory_stats(DIM())
    reset!(tstats, pp)
    @test size(tstats.positions, 2) == 1
    @test length(tstats.dhams) == 1
    reset!(tstats, pp)
    @test size(tstats.positions, 2) == 1
    @test length(tstats.dhams) == 1
    @test tstats.dhams[1] == 0.0
end

@testset "Euclidean phasepoint with non-identity metric" begin
    metric = Diagonal([2.0, 0.5])
    pos = [1.0, 1.0]
    mom = [1.0, 1.0]
    pp = euclidean_phasepoint(pot_f, grad_f, metric, pos, mom)
    expected_kin = 0.5 * (log(2.0) + log(0.5) + 1.0/2.0 + 1.0/0.5)
    expected_pot = 1.0
    @test pp.pot ≈ expected_pot
    @test pp.ham ≈ expected_pot + expected_kin
    @test pp.dham_dmom ≈ metric \ mom
end

@testset "Multiple leapfrog steps preserve Hamiltonian" begin
    rng = Xoshiro(314)
    pp = make_phasepoint(randn(rng, DIM()), randn(rng, DIM()))
    ham_initial = pp.ham
    for _ in 1:100
        leapfrog!(pp; stepsize=0.1)
    end
    @test abs(pp.ham - ham_initial) < 0.1
end

@testset "Generalized leapfrog with Euclidean metric" begin
    pos = [1.0, 2.0]
    mom = [0.5, -0.3]
    pp_lf = make_phasepoint(copy(pos), copy(mom))
    leapfrog!(pp_lf; stepsize=0.1)
    pp_glf = make_phasepoint(copy(pos), copy(mom))
    generalized_leapfrog!(pp_glf; stepsize=0.1, n_fi_steps=5)
    @test pp_lf.pos ≈ pp_glf.pos atol=1e-8
    @test pp_lf.mom ≈ pp_glf.mom atol=1e-8
end

@testset "Implicit midpoint with Euclidean metric" begin
    pos = [1.0, 2.0]
    mom = [0.5, -0.3]
    pp = make_phasepoint(copy(pos), copy(mom))
    ham_before = pp.ham
    implicit_midpoint!(pp; stepsize=0.1, n_fi_steps=5)
    @test abs(pp.ham - ham_before) < 0.01
end

@testset "Dual averaging custom target" begin
    da = dual_averaging_state(1.0; target=0.65)
    initial_step = da.current
    for _ in 1:200
        fit!(da, 0.9)
    end
    @test da.current > initial_step
    @test isfinite(da.current)
    @test da.current > 0
end

@testset "Welford variance reset via new instance" begin
    wv1 = welford_var(DIM())
    step!(wv1, [1.0, 2.0])
    step!(wv1, [3.0, 4.0])
    @test wv1.n ≈ 2.0
    wv2 = welford_var(DIM())
    @test wv2.n ≈ 0.0
    @test all(wv2.mean .== 0.0)
    @test all(wv2.var .== 0.0)
end

@testset "NUTS with non-identity metric" begin
    rng = Xoshiro(999)
    metric = Diagonal([0.5, 2.0])
    pos0 = zeros(DIM())
    mom0 = randn(rng, DIM())
    pp = euclidean_phasepoint(pot_f, grad_f, metric, copy(pos0), copy(mom0))
    state = nuts_state(pp;
        rng=rng,
        step_f=phasepoint -> leapfrog!(phasepoint; stepsize=0.5)
    )
    for _ in 1:10
        @invalidatedependants! state.init.mom = randn(rng, DIM())
        step!(state)
    end
    @test all(isfinite, state.init.pos)
    @test norm(state.init.pos) < 30.0
end

@testset "partial function" begin
    add(a, b; c=0) = a + b + c
    p = partial(add, 10; c=5)
    @test p(3) == 18
    step_f = partial(leapfrog!; stepsize=0.5)
    @test step_f.stepsize == 0.5
    pp = make_phasepoint([1.0, 2.0], [0.5, -0.3])
    ham_before = pp.ham
    step_f(pp)
    @test all(isfinite, pp.pos)
    @test all(isfinite, pp.mom)
    subtract(a, b) = a - b
    sub5 = partial(subtract, :, 5)
    @test sub5(10) == 5
    div3arg(a, b, c) = (a + b) / c
    p2 = partial(div3arg, 1, :, 2)
    @test p2(3) == 2.0
    p3 = partial(identity; alpha=0.1, beta=0.2)
    @test p3.alpha == 0.1
    @test p3.beta == 0.2
end

@testset "hmc_state with stats_f=nothing" begin
    rng = Xoshiro(123)
    pp = make_phasepoint(zeros(DIM()), randn(rng, DIM()))
    state = hmc_state(pp;
        rng=rng,
        n_steps=3,
        step_f=partial(leapfrog!; stepsize=0.3),
        stats_f=nothing
    )
    for _ in 1:5
        @invalidatedependants! state.init.mom = randn(rng, DIM())
        step!(state)
    end
    @test all(isfinite, state.init.pos)
    @test norm(state.init.pos) < 20.0
end

@testset "plain HMC produces valid samples" begin
    n_dim = 2
    inv_vars = ones(n_dim)
    pot_f = x -> 0.5 * dot(x, inv_vars .* x)
    grad_f = x -> (pot_f(x), inv_vars .* x)
    sqrt_metric = ones(n_dim)
    rng = Xoshiro(42)
    pos = zeros(n_dim)
    mom = randn(rng, n_dim)
    fwd_pos = similar(pos)
    fwd_mom = similar(mom)
    grad = similar(pos)
    stepsize = 0.5
    for _ in 1:100
        randn!(rng, mom)
        @. mom = sqrt_metric * mom
        pot0, g0 = grad_f(pos)
        kin0 = 0.5 * dot(mom, mom)
        ham0 = pot0 + kin0
        copy!(fwd_pos, pos); copy!(fwd_mom, mom); copy!(grad, g0)
        for _ in 1:10
            @. fwd_mom -= 0.5 * stepsize * grad
            @. fwd_pos += stepsize * fwd_mom
            _, gn = grad_f(fwd_pos); copy!(grad, gn)
            @. fwd_mom -= 0.5 * stepsize * grad
        end
        pot1 = pot_f(fwd_pos)
        kin1 = 0.5 * dot(fwd_mom, fwd_mom)
        log(rand(rng)) < ham0 - pot1 - kin1 && copy!(pos, fwd_pos)
    end
    @test all(isfinite, pos)
    @test norm(pos) < 10.0
end
