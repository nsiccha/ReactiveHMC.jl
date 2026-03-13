using Test, ReactiveHMC, ReactiveObjects, LinearAlgebra, Random, Statistics, ElasticArrays

# Shared setup: standard quadratic potential (standard normal target)
const DIM = 2

pot_f(x) = 0.5 * dot(x, x)
function grad_f(x)
    (pot_f(x), copy(x))
end

function make_phasepoint(pos, mom; metric=Diagonal(ones(DIM)))
    euclidean_phasepoint(pot_f, grad_f, metric, pos, mom)
end

@testset "ReactiveHMC.jl" begin

    @testset "Euclidean phasepoint fields" begin
        pos = [1.0, 2.0]
        mom = [0.5, -0.3]
        pp = make_phasepoint(pos, mom)

        # Potential: 0.5 * (1^2 + 2^2) = 2.5
        @test pp.pot ≈ 2.5

        # With identity metric, kinetic = 0.5 * dot(mom, mom) = 0.5 * 0.34 = 0.17
        expected_kin = 0.5 * dot(mom, mom)
        @test pp.ham ≈ pp.pot + expected_kin

        # Gradient of potential = position (for quadratic)
        @test pp.dham_dpos ≈ pos

        # dham_dmom = M^{-1} * mom = mom for identity metric
        @test pp.dham_dmom ≈ mom
    end

    @testset "Leapfrog approximate Hamiltonian preservation" begin
        pos = [1.0, 2.0]
        mom = [0.5, -0.3]
        pp = make_phasepoint(pos, mom)
        ham_before = pp.ham

        leapfrog!(pp; stepsize=0.01)

        # For small stepsize, Hamiltonian should be nearly preserved
        @test pp.ham ≈ ham_before atol=1e-4
    end

    @testset "Leapfrog reversibility" begin
        pos = [1.0, 2.0]
        mom = [0.5, -0.3]
        pp = make_phasepoint(copy(pos), copy(mom))

        # Forward step
        leapfrog!(pp; stepsize=0.1)

        # Negate momentum
        @invalidatedependants! pp.mom = -pp.mom

        # Backward step (same stepsize)
        leapfrog!(pp; stepsize=0.1)

        # Negate momentum again to restore original direction
        @invalidatedependants! pp.mom = -pp.mom

        @test pp.pos ≈ pos atol=1e-12
        @test pp.mom ≈ mom atol=1e-12
    end

    @testset "Multistep equivalence" begin
        # Single call with multistep(leapfrog!, n_steps=4) should match 4 individual calls
        pos = [1.0, 2.0]
        mom = [0.5, -0.3]
        stepsize = 0.2

        # Version 1: 4 individual leapfrog steps with stepsize/4
        pp1 = make_phasepoint(copy(pos), copy(mom))
        for _ in 1:4
            leapfrog!(pp1; stepsize=stepsize/4)
        end

        # Version 2: multistep wrapper
        pp2 = make_phasepoint(copy(pos), copy(mom))
        multistep(leapfrog!, pp2; n_steps=4, stepsize=stepsize)

        @test pp1.pos ≈ pp2.pos atol=1e-12
        @test pp1.mom ≈ pp2.mom atol=1e-12

        # Also test the curried form
        pp3 = make_phasepoint(copy(pos), copy(mom))
        step_fn = multistep(leapfrog!; n_steps=4)
        step_fn(pp3; stepsize=stepsize)

        @test pp1.pos ≈ pp3.pos atol=1e-12
    end

    @testset "Welford variance" begin
        rng = Xoshiro(123)
        wv = welford_var(DIM)
        n = 5000
        data = [randn(rng, DIM) for _ in 1:n]

        for x in data
            step!(wv, x)
        end

        expected_mean = mean(data)
        # Welford computes population-style variance; compare to var with corrected=false
        expected_var = var(data; corrected=false)

        @test wv.n ≈ n
        @test wv.mean ≈ expected_mean atol=0.05
        @test wv.var ≈ expected_var atol=0.05
    end

    @testset "Welford variance with matrix input" begin
        rng = Xoshiro(456)
        wv = welford_var(DIM)
        data = randn(rng, DIM, 100)

        step!(wv, data)

        @test wv.n ≈ 100
        @test wv.mean ≈ vec(mean(data; dims=2)) atol=0.2
    end

    @testset "Dual averaging convergence" begin
        da = dual_averaging_state(1.0)

        # Feed a constant acceptance rate above the target (default 0.8)
        # Dual averaging should increase the step size
        for _ in 1:200
            fit!(da, 0.95)
        end
        stepsize_high = da.current

        da2 = dual_averaging_state(1.0)
        # Feed a constant acceptance rate below the target
        for _ in 1:200
            fit!(da2, 0.3)
        end
        stepsize_low = da2.current

        # Higher acceptance => larger step size; lower acceptance => smaller step size
        @test stepsize_high > stepsize_low
        @test isfinite(da.current)
        @test isfinite(da.final)
        @test da.current > 0
        @test da2.current > 0
    end

    @testset "NUTS sampler basic" begin
        rng = Xoshiro(42)
        pos0 = zeros(DIM)
        mom0 = randn(rng, DIM)
        pp = make_phasepoint(copy(pos0), copy(mom0))

        state = nuts_state(pp; rng=rng, step_f=phasepoint -> leapfrog!(phasepoint; stepsize=0.5))

        for _ in 1:20
            @invalidatedependants! state.init.mom = randn(rng, DIM)
            step!(state)
        end

        # After sampling, position should be finite
        @test all(isfinite, state.init.pos)
        # For a standard normal target, samples should be reasonable (not too far from origin)
        @test norm(state.init.pos) < 20.0
    end

    @testset "Trajectory stats recording" begin
        rng = Xoshiro(99)
        pp = make_phasepoint(zeros(DIM), randn(rng, DIM))
        tstats = trajectory_stats(DIM)

        state = nuts_state(pp;
            rng=rng,
            step_f=phasepoint -> leapfrog!(phasepoint; stepsize=0.3),
            stats_f=tstats
        )

        # Run one NUTS step with stats recording
        reset!(tstats, state.init)
        @invalidatedependants! state.init.mom = randn(rng, DIM)
        step!(state)

        # Trajectory stats should have recorded positions
        @test size(tstats.positions, 1) == DIM
        @test size(tstats.positions, 2) >= 1
        @test length(tstats.dhams) >= 1
        @test length(tstats.pots) >= 1
        @test length(tstats.idxs) >= 1
    end

    # Callable struct with .stepsize field, needed by sampling_stats
    struct StepFn; stepsize::Float64; end
    (s::StepFn)(pp) = leapfrog!(pp; stepsize=s.stepsize)

    @testset "Sampling stats accumulation" begin
        rng = Xoshiro(77)
        pp = make_phasepoint(zeros(DIM), randn(rng, DIM))
        tstats = trajectory_stats(DIM)
        dstats = sampling_stats(tstats)
        da = dual_averaging_state(0.5)

        state = nuts_state(pp;
            rng=rng,
            step_f=StepFn(0.5),
            stats_f=tstats
        )

        n_iter = 5
        for _ in 1:n_iter
            reset!(tstats, state.init)
            @invalidatedependants! state.init.mom = randn(rng, DIM)
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
        pp = make_phasepoint([1.0, 2.0], randn(rng, DIM))
        tstats = trajectory_stats(DIM)

        # First reset
        reset!(tstats, pp)
        @test size(tstats.positions, 2) == 1
        @test length(tstats.dhams) == 1

        # Second reset should clear and start fresh
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

        # Kinetic = 0.5 * (logdet(metric) + mom' * inv(metric) * mom)
        # = 0.5 * (log(2) + log(0.5) + 1/2 + 1/0.5)
        # = 0.5 * (0 + 0.5 + 2) = 1.25
        expected_kin = 0.5 * (log(2.0) + log(0.5) + 1.0/2.0 + 1.0/0.5)
        expected_pot = 1.0

        @test pp.pot ≈ expected_pot
        @test pp.ham ≈ expected_pot + expected_kin

        # dham_dmom = M^{-1} * mom
        @test pp.dham_dmom ≈ metric \ mom
    end

    @testset "Multiple leapfrog steps preserve Hamiltonian (symplectic)" begin
        rng = Xoshiro(314)
        pp = make_phasepoint(randn(rng, DIM), randn(rng, DIM))
        ham_initial = pp.ham

        # Many steps with moderate stepsize
        for _ in 1:100
            leapfrog!(pp; stepsize=0.1)
        end

        # Symplectic integrator: Hamiltonian error should stay bounded, not grow
        @test abs(pp.ham - ham_initial) < 0.1
    end

end
