# ReactiveHMC.jl

[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://nsiccha.github.io/ReactiveHMC.jl/dev/)
[![CI](https://github.com/nsiccha/ReactiveHMC.jl/actions/workflows/test.yml/badge.svg)](https://github.com/nsiccha/ReactiveHMC.jl/actions/workflows/test.yml)

Hamiltonian Monte Carlo implementation built on [ReactiveObjects.jl](https://github.com/nsiccha/ReactiveObjects.jl).

## Features

- **Multiple phase point types**: Euclidean, Riemannian, SoftAbs, and relativistic geometries
- **Integrators**: leapfrog, generalized leapfrog, and implicit midpoint
- **NUTS sampler**: No-U-Turn Sampler with dynamic trajectory length
- **Adaptation**: step size and mass matrix tuning

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/nsiccha/ReactiveHMC.jl")
```

## Related packages

- [ReactiveObjects.jl](https://github.com/nsiccha/ReactiveObjects.jl) -- reactive kernel framework powering the phase point computations
- [WarmupHMC.jl](https://github.com/nsiccha/WarmupHMC.jl) -- high-level sampling interface built on ReactiveHMC.jl
- [Treebars.jl](https://github.com/nsiccha/Treebars.jl) -- tree-structured progress bars for sampling runs
