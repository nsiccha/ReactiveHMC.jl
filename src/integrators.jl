leapfrog!(phasepoint; stepsize) = begin 
    @. phasepoint.mom -= .5 * stepsize * phasepoint.dham_dpos
    @. phasepoint.pos += stepsize * phasepoint.dham_dmom
    @. phasepoint.mom -= .5 * stepsize * phasepoint.dham_dpos
end
generalized_leapfrog!(phasepoint; stepsize, n_fi_steps) = begin 
    pos0, mom0 = map(copy, (phasepoint.pos, phasepoint.mom))
    for _ in 1:n_fi_steps 
        @. phasepoint.mom = mom0 - .5 * stepsize * phasepoint.dham_dpos
    end
    dham_dmom0 = copy(phasepoint.dham_dmom)
    for _ in 1:n_fi_steps 
        @. phasepoint.pos = pos0 + .5 * stepsize * (dham_dmom0 + phasepoint.dham_dmom)
    end
    @. phasepoint.mom -= .5 * stepsize * phasepoint.dham_dpos
end
implicit_midpoint!(phasepoint; stepsize, n_fi_steps) = begin 
    pos0, mom0 = map(copy, (phasepoint.pos, phasepoint.mom))
    for _ in 1:n_fi_steps 
        (;dham_dmom, dham_dpos) = phasepoint
        @. phasepoint.pos = pos0 + .5 * stepsize * dham_dmom
        @. phasepoint.mom = mom0 - .5 * stepsize * dham_dpos
    end
    @. phasepoint.pos = 2 * phasepoint.pos - pos0
    @. phasepoint.mom = 2 * phasepoint.mom - mom0
end
multistep(f, args...; n_steps, stepsize, kwargs...) = for _ in 1:n_steps
    f(args...; stepsize=stepsize/n_steps, kwargs...)
end
multistep(f; n_steps) = (args...; kwargs...)->multistep(f, args...; n_steps, kwargs...)