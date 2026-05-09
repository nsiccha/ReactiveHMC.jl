module ReactiveHMCWeb

using HTMXObjects
using ReactiveHMC
using ReactiveObjects
using LinearAlgebra
using Random
using Chairmarks, Statistics
using LogDensityProblems
import AdvancedHMC
import DynamicHMC
import NUTS as NUTSjl
using AlgebraOfVega
using DataFrames
using TestModules

include("test/runtests.jl")

# fmt_time from HTMXObjects

# ── Diagonal MVN target ──────────────────────────────────────────────────────

function make_diagonal_mvn(; n_dim=2, condition_number=1.0)
    vars = n_dim == 1 ? [1.0] : [condition_number^((i-1)/(n_dim-1)) for i in 1:n_dim]
    inv_vars = 1.0 ./ vars
    pot_f = x -> 0.5 * dot(x, inv_vars .* x)
    grad_f = x -> (pot_f(x), inv_vars .* x)
    logdensity_f = x -> -0.5 * dot(x, inv_vars .* x)
    logdensity_and_gradient_f = x -> (-0.5 * dot(x, inv_vars .* x), -(inv_vars .* x))
    (; pot_f, grad_f, logdensity_f, logdensity_and_gradient_f, vars, inv_vars, n_dim)
end

# LogDensityProblems wrapper for AdvancedHMC / DynamicHMC
struct DiagonalMVNTarget
    n_dim::Int
    inv_vars::Vector{Float64}
end
LogDensityProblems.capabilities(::Type{DiagonalMVNTarget}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(t::DiagonalMVNTarget) = t.n_dim
function LogDensityProblems.logdensity_and_gradient(t::DiagonalMVNTarget, x::AbstractVector)
    (-0.5 * dot(x, t.inv_vars .* x), -(t.inv_vars .* x))
end

# NUTS.jl LogDensity adapter
struct NUTSjlTarget
    inv_vars::Vector{Float64}
end

NUTSjl.log_density_gradient!(t::NUTSjlTarget, x::AbstractVector, g::AbstractVector) = begin
    @. g = -(t.inv_vars * x)
    -0.5 * dot(x, t.inv_vars .* x)
end

# ── Result row helper ────────────────────────────────────────────────────────

function make_result(; name, trial, n_grads, n_dim, condition_number, stepsize)
    (; name, trial, n_grads, n_dim, condition_number, stepsize)
end

# ── Aggregation + rendering helpers ──────────────────────────────────────────

function collect_sweep(app; dims=[2, 4, 8, 16, 32, 64, 128], kappas=[1.0, 100.0], stepsize=0.5, n_steps=10, seed=42)
    rows = NamedTuple[]
    for n_dim in dims, kappa in kappas
        for r in @memo app.bench_result(; n_dim, condition_number=kappa, stepsize, n_steps, seed).rows
            push!(rows, r)
        end
    end
    rows
end

function results_to_dataframe(rows)
    df = DataFrame(
        method = String[],
        dim = Int[],
        kappa = Float64[],
        stepsize = Float64[],
        grads_per_step = Float64[],
        time_us = Float64[],
        grads_per_sec = Float64[],
        overhead_vs_plain = Float64[],
        us_per_grad = Float64[],
        allocs = Float64[],
        bytes = Float64[],
    )
    # Index Plain HMC us_per_grad per (dim, kappa) for relative overhead
    plain_us_per_grad = Dict{Tuple{Int,Float64}, Float64}()
    for r in rows
        r.name == "Plain HMC" || continue
        plain_us_per_grad[(r.n_dim, r.condition_number)] = median(r.trial).time * 1e6 / r.n_grads
    end
    for r in rows
        med = median(r.trial)
        us_pg = med.time * 1e6 / r.n_grads
        plain_uspg = get(plain_us_per_grad, (r.n_dim, r.condition_number), us_pg)
        push!(df, (
            r.name, r.n_dim, r.condition_number, r.stepsize,
            r.n_grads,
            med.time * 1e6,
            r.n_grads / med.time,
            us_pg / plain_uspg,
            us_pg,
            Float64(med.allocs),
            Float64(med.bytes),
        ))
    end
    df
end

# ── HTML rendering ───────────────────────────────────────────────────────────

function render_comparison_table(results)
    rows = map(results) do r
        med = median(r.trial)
        mn = minimum(r.trial)
        h.tr(
            h.td(r.name),
            h.td(string(round(r.n_grads; digits=1))),
            h.td(fmt_time(med.time)),
            h.td(fmt_time(mn.time)),
            h.td(string(round(r.n_grads / med.time; sigdigits=3))),
            h.td(string(med.allocs)),
            h.td(Base.format_bytes(med.bytes)),
            h.td(string(length(r.trial.samples))),
        )
    end
    h.table(class="striped rhmc-bench-table")(
        h.thead(h.tr(
            h.th("Method"),
            h.th("Grads/step"),
            h.th("Median time"),
            h.th("Min time"),
            h.th("Grads/sec (med)"),
            h.th("Allocs"),
            h.th("Bytes"),
            h.th("Bench samples"),
        )),
        h.tbody(rows...),
    )
end

# ── App ──────────────────────────────────────────────────────────────────────

CSS = """
    .rhmc-bench-table{font-size:.85rem}
"""

@htmx struct AppContext

    cache_path = joinpath(dirname(dirname(@__DIR__)), "web", "cache")

    # One configuration → eight benchmark variants. Named siblings own the
    # per-method computation; the `bench(method)` dispatcher does the
    # canonical `getproperty(__parent__, method)` lookup (mirrors WHMC's
    # `result(method)` shape — see the `do` skill §4).
    @struct bench_result(; n_dim, condition_number, stepsize, n_steps, seed) = begin

        @struct rhmc_nuts_no_stats = begin
            @cached row = let
                target = make_diagonal_mvn(; n_dim, condition_number)
                rng = Random.Xoshiro(seed)
                metric = Diagonal(ones(n_dim))
                pp = euclidean_phasepoint(target.pot_f, target.grad_f, metric, zeros(n_dim), randn(rng, n_dim))
                step_f = partial(leapfrog!; stepsize)
                state = nuts_state(pp; rng, step_f, stats_f=nothing)

                do_step! = () -> begin
                    @invalidatedependants! state.init.mom = randn(rng, n_dim)
                    step!(state)
                end

                # Warmup
                do_step!()

                # Separate counting pass with stats to derive mean grads/step.
                rng2 = Random.Xoshiro(seed)
                pp2 = euclidean_phasepoint(target.pot_f, target.grad_f, metric, zeros(n_dim), randn(rng2, n_dim))
                ts = trajectory_stats(n_dim)
                state2 = nuts_state(pp2; rng=rng2, step_f, stats_f=ts)
                n_count = 50
                total_grads = 0
                for _ in 1:n_count
                    reset!(ts, state2.init)
                    @invalidatedependants! state2.init.mom = randn(rng2, n_dim)
                    step!(state2)
                    total_grads += length(ts.dhams) - 1
                end
                # Run the timed steps without stats so the trial reflects the
                # no-stats configuration.
                for _ in 1:n_count
                    do_step!()
                end
                mean_grads = total_grads / n_count

                trial = @be do_step!()
                make_result(; name="ReactiveHMC NUTS (no stats)", trial,
                              n_grads=mean_grads, n_dim, condition_number, stepsize)
            end
        end

        @struct rhmc_nuts_full = begin
            @cached row = let
                target = make_diagonal_mvn(; n_dim, condition_number)
                rng = Random.Xoshiro(seed)
                metric = Diagonal(ones(n_dim))
                pp = euclidean_phasepoint(target.pot_f, target.grad_f, metric, zeros(n_dim), randn(rng, n_dim))
                step_f = partial(leapfrog!; stepsize)
                stats_f = trajectory_stats(n_dim)
                state = nuts_state(pp; rng, step_f, stats_f)

                do_step! = () -> begin
                    reset!(stats_f, state.init)
                    @invalidatedependants! state.init.mom = randn(rng, n_dim)
                    step!(state)
                end

                # Warmup
                do_step!()

                n_count = 50
                total_grads = 0
                for _ in 1:n_count
                    do_step!()
                    total_grads += length(stats_f.dhams) - 1
                end
                mean_grads = total_grads / n_count

                trial = @be do_step!()
                make_result(; name="ReactiveHMC NUTS (stats)", trial,
                              n_grads=mean_grads, n_dim, condition_number, stepsize)
            end
        end

        @struct rhmc_hmc_no_stats = begin
            @cached row = let
                target = make_diagonal_mvn(; n_dim, condition_number)
                rng = Random.Xoshiro(seed)
                metric = Diagonal(ones(n_dim))
                pp = euclidean_phasepoint(target.pot_f, target.grad_f, metric, zeros(n_dim), randn(rng, n_dim))
                step_f = partial(leapfrog!; stepsize)
                state = hmc_state(pp; rng, step_f, stats_f=nothing, n_steps)

                do_step! = () -> ReactiveHMC.step!(state)

                # Warmup
                do_step!()

                trial = @be do_step!()
                make_result(; name="ReactiveHMC HMC (no stats)", trial,
                              n_grads=Float64(n_steps), n_dim, condition_number, stepsize)
            end
        end

        @struct rhmc_hmc_full = begin
            @cached row = let
                target = make_diagonal_mvn(; n_dim, condition_number)
                rng = Random.Xoshiro(seed)
                metric = Diagonal(ones(n_dim))
                pp = euclidean_phasepoint(target.pot_f, target.grad_f, metric, zeros(n_dim), randn(rng, n_dim))
                step_f = partial(leapfrog!; stepsize)
                stats_f = trajectory_stats(n_dim)
                state = hmc_state(pp; rng, step_f, stats_f, n_steps)

                do_step! = () -> ReactiveHMC.step!(state)

                # Warmup
                do_step!()

                trial = @be do_step!()
                make_result(; name="ReactiveHMC HMC (stats)", trial,
                              n_grads=Float64(n_steps), n_dim, condition_number, stepsize)
            end
        end

        @struct plain_hmc = begin
            @cached row = let
                target = make_diagonal_mvn(; n_dim, condition_number)
                rng = Random.Xoshiro(seed)
                metric = ones(n_dim)  # diagonal metric as vector
                sqrt_metric = sqrt.(metric)
                inv_metric = 1.0 ./ metric

                pos = zeros(n_dim)
                mom = randn(rng, n_dim)

                # Pre-allocate workspace
                fwd_pos = similar(pos)
                fwd_mom = similar(mom)
                grad = similar(pos)

                plain_leapfrog! = (pos, mom, grad, stepsize, inv_metric) -> begin
                    @. mom -= 0.5 * stepsize * grad
                    @. pos += stepsize * inv_metric * mom
                    _, grad_new = target.grad_f(pos)
                    copy!(grad, grad_new)
                    @. mom -= 0.5 * stepsize * grad
                end

                plain_hmc_step! = (pos, mom, fwd_pos, fwd_mom, grad) -> begin
                    randn!(rng, mom)
                    @. mom = sqrt_metric * mom

                    pot0, g0 = target.grad_f(pos)
                    kin0 = 0.5 * dot(mom, inv_metric .* mom)
                    ham0 = pot0 + kin0

                    copy!(fwd_pos, pos)
                    copy!(fwd_mom, mom)
                    copy!(grad, g0)

                    for _ in 1:n_steps
                        plain_leapfrog!(fwd_pos, fwd_mom, grad, stepsize, inv_metric)
                    end

                    pot1 = target.pot_f(fwd_pos)
                    kin1 = 0.5 * dot(fwd_mom, inv_metric .* fwd_mom)
                    ham1 = pot1 + kin1
                    dham = ham0 - ham1

                    if log(rand(rng)) < dham
                        copy!(pos, fwd_pos)
                    end
                end

                do_step! = () -> plain_hmc_step!(pos, mom, fwd_pos, fwd_mom, grad)

                # Warmup
                do_step!()

                trial = @be do_step!()
                make_result(; name="Plain HMC", trial,
                              n_grads=Float64(n_steps), n_dim, condition_number, stepsize)
            end
        end

        @struct advancedhmc = begin
            @cached row = let
                target = make_diagonal_mvn(; n_dim, condition_number)
                rng = Random.Xoshiro(seed)
                metric = AdvancedHMC.DiagEuclideanMetric(n_dim)
                hamiltonian = AdvancedHMC.Hamiltonian(metric, target.logdensity_f,
                    (θ) -> target.logdensity_and_gradient_f(θ))
                integrator = AdvancedHMC.Leapfrog(stepsize)
                term = AdvancedHMC.StrictGeneralisedNoUTurn()
                trajectory = AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator, term)

                θ = zeros(n_dim)
                z = AdvancedHMC.phasepoint(rng, θ, hamiltonian)

                do_step! = () -> begin
                    z = AdvancedHMC.phasepoint(rng, z.θ, hamiltonian)
                    trans = AdvancedHMC.transition(rng, hamiltonian, trajectory, z)
                    z = trans.z
                end

                # Warmup
                do_step!()

                # Count average gradient evals
                n_count = 50
                total_grads = 0
                for _ in 1:n_count
                    z = AdvancedHMC.phasepoint(rng, z.θ, hamiltonian)
                    trans = AdvancedHMC.transition(rng, hamiltonian, trajectory, z)
                    z = trans.z
                    total_grads += trans.stat.n_steps
                end
                mean_grads = total_grads / n_count

                trial = @be do_step!()
                make_result(; name="AdvancedHMC NUTS", trial,
                              n_grads=mean_grads, n_dim, condition_number, stepsize)
            end
        end

        @struct dynamichmc = begin
            @cached row = let
                target = DiagonalMVNTarget(n_dim, make_diagonal_mvn(; n_dim, condition_number).inv_vars)
                rng = Random.Xoshiro(seed)
                κ = DynamicHMC.GaussianKineticEnergy(n_dim)
                H = DynamicHMC.Hamiltonian(κ, target)
                algorithm = DynamicHMC.NUTS()

                Q = DynamicHMC.evaluate_ℓ(target, zeros(n_dim))

                do_step! = () -> begin
                    Q, _ = DynamicHMC.sample_tree(rng, algorithm, H, Q, stepsize)
                end

                # Warmup
                do_step!()

                # Count average gradient evals
                n_count = 50
                total_grads = 0
                for _ in 1:n_count
                    Q, stats = DynamicHMC.sample_tree(rng, algorithm, H, Q, stepsize)
                    total_grads += stats.steps
                end
                mean_grads = total_grads / n_count

                trial = @be do_step!()
                make_result(; name="DynamicHMC NUTS", trial,
                              n_grads=mean_grads, n_dim, condition_number, stepsize)
            end
        end

        @struct nutsjl = begin
            @cached row = let
                target = make_diagonal_mvn(; n_dim, condition_number)
                rng = Random.Xoshiro(seed)
                posterior = NUTSjlTarget(target.inv_vars)

                state = (; rng, posterior, stepsize, position=zeros(n_dim))

                # Warmup
                state = NUTSjl.nuts!!(state)

                # Count average gradient evals
                n_count = 50
                total_grads = 0
                for _ in 1:n_count
                    state = NUTSjl.nuts!!(state)
                    total_grads += state.n_leapfrog
                end
                mean_grads = total_grads / n_count

                do_step! = () -> begin
                    state = NUTSjl.nuts!!(state)
                end

                trial = @be do_step!()
                make_result(; name="NUTS.jl", trial,
                              n_grads=mean_grads, n_dim, condition_number, stepsize)
            end
        end

        # Dispatcher — bare `getproperty(__parent__, method)` lookup, no
        # branching, no name munging. (`do` skill §4.)
        @struct bench(method::Symbol) = begin
            row = getproperty(__parent__, method).row
        end

        rows = [
            bench(:rhmc_nuts_no_stats).row,
            bench(:rhmc_nuts_full).row,
            bench(:rhmc_hmc_no_stats).row,
            bench(:rhmc_hmc_full).row,
            bench(:plain_hmc).row,
            bench(:nutsjl).row,
            bench(:advancedhmc).row,
            bench(:dynamichmc).row,
        ]
    end

    page(content) = htmx(
        h.body(h.main(class="container")(content));
        pico_version="2",
        extra_head=(
            vega_head()...,
            h.style(CSS),
            h.title("ReactiveHMC Benchmark"),
        ),
    )

    @get index(; stepsize::Float64=0.5, n_steps::Int=10, seed::Int=42) = begin
        rows = collect_sweep(__self__; stepsize, n_steps, seed)
        tbl = results_to_dataframe(rows)
        page[h.div(
            h.h1("ReactiveHMC — Benchmark Explorer"),
            explorer_widget(Dict("benchmark" => tbl);
                default_ds="benchmark",
                title="Benchmark Results",
                default_x="dim", default_y="overhead_vs_plain",
                default_color="method", default_col="kappa",
                default_mark="line",
                default_log_x=true, default_log_y=true,
            ),
            h.hr(),
            h.p(h.a(href="/table?dim=10")("Single-config table view")),
        )]
    end

    @get clear_cache = begin
        rm(cache_path; recursive=true, force=true)
        page[h.div(
            h.h1("Cache cleared"),
            h.p(h.a(href="/")("Back to explorer")),
        )]
    end

    @get debug(; stepsize::Float64=0.5, n_steps::Int=10, seed::Int=42) = begin
        rows = collect_sweep(__self__; stepsize, n_steps, seed)
        r1 = rows[1]
        s1 = r1.trial.samples[1]
        lines = [
            "rows: $(length(rows))",
            "r1.name: $(r1.name)",
            "r1.trial samples: $(length(r1.trial.samples))",
            "s1: $(s1)",
            "s1.warmup: $(s1.warmup) ($(typeof(s1.warmup)))",
            "s1.time: $(s1.time)",
            "s1.allocs: $(s1.allocs)",
            "",
            "DataFrame test:",
        ]
        tbl = results_to_dataframe(rows)
        push!(lines, "nrow: $(nrow(tbl))")
        push!(lines, "ncol: $(ncol(tbl))")
        if nrow(tbl) > 0
            push!(lines, string(first(tbl, 5)))
        end
        join(lines, "\n")
    end

    @get table(; dim::Int=10, kappa::Float64=100.0, stepsize::Float64=0.5, n_steps::Int=10, seed::Int=42) = begin
        results = @memo bench_result(; n_dim=dim, condition_number=kappa, stepsize, n_steps, seed).rows
        page[h.div(
            h.h1("ReactiveHMC.jl — Benchmark Comparison"),
            h.p("Diagonal MVN: $(dim)D, κ=$(kappa), stepsize=$(stepsize)"),
            render_comparison_table(results),
            h.p(h.a(href="/")("Back to explorer")),
        )]
    end

    @get table_sweep(; stepsize::Float64=0.5, n_steps::Int=10, seed::Int=42) = begin
        rows = collect_sweep(__self__; stepsize, n_steps, seed)
        tbl = results_to_dataframe(rows)
        page[h.div(
            h.h1("Sweep Data — $(nrow(tbl)) rows"),
            h.p("$(length(rows)) benchmark configs × samples each"),
            h.details(h.summary("First 50 rows"))(
                h.pre(string(first(tbl, 50))),
            ),
            h.p(h.a(href="/")("Back to explorer")),
        )]
    end

    @include tests = TestRoutes(; __req__, test_module=@__MODULE__)
    @include structure = StructureRoutes(; root=AppContext)
end

function __init__()
    route!(AppContext())
end

end # module ReactiveHMCWeb
