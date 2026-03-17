module ReactiveHMCTests
using Test, Random, ReactiveHMC, ReactiveObjects, LinearAlgebra, Statistics, ElasticArrays, TestModules
include("ReactiveHMCTests.jl")
end

using TestModules
runtests!(ReactiveHMCTests)
