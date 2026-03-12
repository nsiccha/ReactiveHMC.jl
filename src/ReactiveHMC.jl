module ReactiveHMC

using ReactiveObjects, LinearAlgebra, LogExpFunctions, Random
using ElasticArrays

import ReactiveObjects: rcopy!

export leapfrog!, generalized_leapfrog!, implicit_midpoint!, multistep
export euclidean_phasepoint, riemannian_phasepoint, riemannian_softabs_phasepoint, relativistic_euclidean_phasepoint, relativistic_riemannian_phasepoint, relativistic_riemannian_softabs_phasepoint
export nuts_state, step!
export dual_averaging_state, welford_var, fit!
export trajectory_stats, sampling_stats

include("integrators.jl")
include("phasepoints.jl")
include("energies.jl")
include("samplers.jl")
include("adaptation.jl")
include("statistics.jl")

end # module ReactiveHMC
