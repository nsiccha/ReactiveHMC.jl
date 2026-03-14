"""
    partial(f, args...; kwargs...)
    partial(f, :, args...; kwargs...)
    partial(f, l1, :, args...; kwargs...)

Create a partially applied function. Keyword arguments are accessible as properties
(e.g. `partial(leapfrog!; stepsize=0.5).stepsize`).

# Examples
```julia
step_f = partial(leapfrog!; stepsize=0.5)
step_f(phasepoint)  # calls leapfrog!(phasepoint; stepsize=0.5)
step_f.stepsize      # 0.5
```
"""
struct PartialFunction{F<:Union{Function,Type},L<:Tuple,R<:Tuple,K<:NamedTuple} <: Function
    func::F
    largs::L
    rargs::R
    kwargs::K
end

(f::PartialFunction)(args...; kwargs...) = _func(f)(_largs(f)..., args..., _rargs(f)...; _kwargs(f)..., kwargs...)

_func(f::PartialFunction) = getfield(f, :func)
_largs(f::PartialFunction) = getfield(f, :largs)
_rargs(f::PartialFunction) = getfield(f, :rargs)
_kwargs(f::PartialFunction) = getfield(f, :kwargs)

Base.getproperty(f::PartialFunction, x::Symbol) = getfield(_kwargs(f), x)

function Base.show(io::IO, f::PartialFunction)
    print(io, _func(f), "(")
    join(io, filter(x -> isa(x, Union{Number,Symbol,AbstractString}), _largs(f)), ", ")
    kws = ["$k=$v" for (k, v) in pairs(filter(x -> isa(x, Union{Number,Symbol,AbstractString}), _kwargs(f)))]
    if !isempty(kws)
        print(io, "; ")
        join(io, kws, ", ")
    end
    print(io, ")")
end

partial(f, args...; kwargs...) = PartialFunction(f, args, (), (; kwargs...))
partial(f, ::Colon, args...; kwargs...) = PartialFunction(f, (), args, (; kwargs...))
partial(f, l1, ::Colon, args...; kwargs...) = PartialFunction(f, (l1,), args, (; kwargs...))
