"""
    dual_averaging_state(init; target=0.8, ...)

Step size adaptation via Nesterov-style dual averaging.
Call `fit!(da_state, acceptance_rate)` after each NUTS step during warmup.
Use `da_state.current` during adaptation and `da_state.final` after warmup ends.
When changing the metric, create a fresh `da_state` seeded with the current step size.
"""
@reactive dual_averaging_state(init; target=.8, regularization_scale=.05, relaxation_exponent=.75, offset=10) = begin
    m = one(init)
    H = zero(init)
    mu = log(10) + log(init)
    log_current = mu - sqrt(m) / regularization_scale * H
    log_final = zero(init)
    current = exp(log_current)
    final = exp(log_final)
    fit!(x) = begin 
        m += 1 
        H += (target - x - H) / (m + offset)
        log_final += m^(-relaxation_exponent) * (log_current - log_final)
    end
end

smooth(prev, new, new_weight) = (1-new_weight)*prev + new_weight*new

"""
    welford_var(dim)

Online variance estimation using Welford's algorithm. Feed samples via `step!(wv, x)`.
Fields: `.n` (count), `.mean` (running mean), `.var` (running variance).
Accepts vectors or matrices (columns as samples).

For metric adaptation:
- Stan-style: `Diagonal(max.(1e-6, wv.var))` from position samples
- Nutpie-style: `Diagonal(max.(1e-6, sqrt.(wvp.var ./ wvg.var)))` from pos + grad
"""
@reactive welford_var(dim) = begin
    n = 0.
    mean = zeros(dim)
    var = zeros(dim)
    step!(x::AbstractVector; dn=1.) = begin 
        n += dn
        w = dn / n
        @. var = smooth(var, (x - smooth(mean, x, w)) * (x - mean), w)
        @. mean = smooth(mean, x, w)
    end
    step!(x::AbstractMatrix; kwargs...) = for xi in eachcol(x)
        step!(__self__, xi; kwargs...)
    end
end