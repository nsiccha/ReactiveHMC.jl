# ReactiveHMC.jl

Hamiltonian Monte Carlo built on [ReactiveObjects.jl](https://github.com/nsiccha/ReactiveObjects.jl) — phase points, integrators, and samplers that avoid redundant computation through reactive dependency tracking.

## Quick start

```julia
using ReactiveHMC, LinearAlgebra, Random

# Define your target distribution
pot_f(pos) = 0.5 * sum(abs2, pos)
grad_f(pos) = (pot_f(pos), +pos)

dim = 10
pos = randn(dim)
mom = randn(dim)

# Create a reactive phase point
pp = euclidean_phasepoint(pot_f, grad_f, Diagonal(ones(dim)), pos, mom)

# Integrate with leapfrog
leapfrog!(pp; stepsize=0.1)

# Only momentum changed → position-dependent quantities were NOT recomputed
# Now accessing pp.dham_dpos triggers recomputation of gradients
```

## Phase points

Phase points are `@reactive` kernels tracking position, momentum, potential energy, gradients, metric, kinetic energy, and the Hamiltonian. When only momentum changes (as in the momentum half-step of leapfrog), position-dependent quantities like gradients and the metric are not recomputed.

### Euclidean metric

```julia
euclidean_phasepoint(pot_f, grad_f, metric, pos, mom)
```

Standard HMC with a fixed mass matrix. `pot_f(pos)` returns the potential, `grad_f(pos)` returns `(pot, dpot_dpos)`.

### Riemannian metric

```julia
riemannian_phasepoint(pot_f, grad_f, metric_f, metric_grad_f, pos, mom)
```

Position-dependent metric. `metric_f(pos)` returns `(pot, dpot_dpos, metric)`, `metric_grad_f(pos)` returns all of those plus `metric_grad` (a 3D tensor of metric derivatives).

### SoftAbs Riemannian metric

```julia
riemannian_softabs_phasepoint(pot_f, grad_f, premetric_f, premetric_grad_f, pos, mom; alpha=20.0)
```

SoftAbs transformation of the Hessian for a positive-definite metric. `alpha` controls the sharpness of the absolute value approximation.

### Relativistic variants

Each metric type has a relativistic counterpart with additional `c` (speed of light) and `m` (mass) parameters:

- `relativistic_euclidean_phasepoint`
- `relativistic_riemannian_phasepoint`
- `relativistic_riemannian_softabs_phasepoint`

## Integrators

All integrators operate on phase points in-place:

```julia
leapfrog!(phasepoint; stepsize)
```

Standard leapfrog (Störmer-Verlet). Requires Euclidean metric.

```julia
generalized_leapfrog!(phasepoint; stepsize, n_fi_steps)
```

Generalized leapfrog with fixed-point iterations for position-dependent metrics. `n_fi_steps` controls the number of implicit solve iterations.

```julia
implicit_midpoint!(phasepoint; stepsize, n_fi_steps)
```

Implicit midpoint integrator.

```julia
multistep(integrator; n_steps)
```

Wraps any integrator to take `n_steps` sub-steps per call (dividing `stepsize` accordingly). Can be curried:

```julia
stepper = multistep(generalized_leapfrog!; n_steps=10)
stepper(phasepoint; stepsize=1.0, n_fi_steps=5)
```

## Samplers

### NUTS

```julia
state = nuts_state(phasepoint; rng, step_f, stats_f=nothing, max_depth=10, min_dham=-1000.0)
```

No-U-Turn Sampler with multinomial sampling. `step_f` is the integrator, `stats_f` is an optional trajectory recorder.

Call `ReactiveHMC.step!(state)` to advance. The accepted sample is in `state.init.pos`.

### Basic HMC

```julia
state = hmc_state(phasepoint; rng, n_steps=1, step_f, stats_f)
```

Fixed-trajectory-length HMC.

## Adaptation

### Step size (dual averaging)

```julia
da = dual_averaging_state(initial_stepsize; target=0.8)
# After each NUTS step:
fit!(da, acceptance_rate)
# During warmup use da.current, after warmup use da.final
```

### Mass matrix (Welford variance)

```julia
wv = welford_var(dim)
# Feed samples:
step!(wv, position_vector)
# Use wv.var for diagonal mass matrix adaptation
```

## Statistics

### Trajectory statistics

```julia
tstats = trajectory_stats(dim)
```

Records positions, gradients, potentials, and Hamiltonian errors during NUTS tree expansion. Use as `stats_f` in `nuts_state`. Call `reset!(tstats, phasepoint)` before each step.

### Sampling statistics

```julia
dstats = sampling_stats(tstats)
```

Accumulates per-iteration statistics across a full sampling run: `draws`, `n_steps`, `stepsizes`, `acc_rate`, `diverged`, plus full trajectory history for visualization.
