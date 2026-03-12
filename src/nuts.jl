fillf(f::Function, value, n::Int) = [f(value) for _ in 1:n]
finiteorneginf(x) = isfinite(x) ? x : typeof(x)(-Inf)
min1exp(x) = x >= 0 ? one(x) : exp(x)
badd(args...) = Base.broadcasted(+, args...)
randbernoullilog(rng, logprob) = logprob > 0 ? true : -randexp(rng) < logprob 
logswapprob(tree) = tree.log_weight[1] - tree.log_weight[2]
compute_criterion(mom, bwd_dham_dmom, fwd_dham_dmom) = (dot(mom, bwd_dham_dmom) > 0 && dot(mom, fwd_dham_dmom) > 0)

trajectory(d::Int) = trajectory(zeros(d), zeros(d))
trajectory(bwd, fwd) = (;bwd, fwd)
mv(mom, dham_dmom) = (;mom, dham_dmom)
mv(d::Int) = mv(zeros(d), zeros(d))
tree(d::Int) = (;
    log_weight=fill(-Inf, 2),
    bwd=mv(d),
    bwd_fwd=mv(d),
    summed_mom=trajectory(d),
)
tree(phasepoint) = tree(length(phasepoint.pos))

@reactive nuts_state(
    init;
    rng,
    max_depth=10,
    min_dham=-1000.,
    step_f=nothing,
    stats_f=nothing
) = begin 
    gofwd = true
    may_sample = true 
    may_continue = true
    fwdbwd = fillf(deepcopy, init, 2)
    fwd = Ref(fwdbwd[gofwd ? 1 : 2])
    bwd = Ref(fwdbwd[gofwd ? 2 : 1])
    trees = fillf(tree, init, max_depth+1)
    proposals = fillf(deepcopy, init, max_depth+2)
    # To have a reactive dham, init.ham and fwd.ham have to trigger invalidation of dham.
    # To do this automatically and in a general way, fwd (and actually also init)
    # will have to know that they are a `(obj::nuts_state).fwd` property,
    # i.e. they have to know the type of their parent, their own name (to trigger the right invalidations),
    # and have to (at least) have access to their parent's `valid` vector (to able to actually invalidate).
    # In general however, only having access to that vector is not enough, as the parent itself may 
    # have to invalidate its parent. However, these extra `valid` vectors can probably be accumulated recursively
    # in a Tuple. 
    dham = 0. #finiteorneginf(init.ham - fwd.ham)
    diverged = !(dham >= min_dham)
    stepfwd!() = step_f(fwd)
    collectstats!() = isnothing(stats_f) || stats_f(__self__)
    logadvanceprob(depth) = trees[depth-1].log_weight[1] - trees[depth].log_weight[1]
    swapproposal!(i, j=length(proposals)) = begin 
        proposals[i], proposals[j] = proposals[j], proposals[i]
    end
    step!(;force=true) = begin 
        ReactiveHMC.ReactiveObjects.restore!(__self__; force)
        bwd.mom .*= -1
        trees[1].log_weight[1] = 0.
        for depth in 1:max_depth
            rand(rng, Bool) && flip!(__self__, depth)
            finish_tree!(__self__, depth)
            may_sample || break
            randbernoullilog(rng, logswapprob(trees[depth])) && swapproposal!(__self__, depth)
            may_continue || break
        end
        rcopy!(init, proposals[end])
    end
    flip!(depth) = if depth > 1 
        gofwd = !gofwd
        tree = trees[depth]
        @. tree.bwd.mom = -bwd.mom
        @. tree.bwd.dham_dmom = -bwd.dham_dmom
        @. tree.summed_mom.fwd *= -1
    end 
    finish_tree!(depth) = begin 
        tree = trees[depth]
        suptree = trees[depth+1]
        tree.log_weight[2] = tree.log_weight[1]
        if depth == 1
            rcopy!(suptree.bwd, (;fwd.mom, fwd.dham_dmom))
        else
            rcopy!(suptree.bwd, tree.bwd)
            rcopy!(tree.bwd_fwd, (;fwd.mom, fwd.dham_dmom))
            tree.summed_mom.bwd .= tree.summed_mom.fwd
        end
        start_tree!(__self__, depth)
        may_continue || return may_sample = false
        suptree.log_weight[1] = logaddexp(tree.log_weight[1], tree.log_weight[2])
        may_continue = if depth == 1
            suptree.summed_mom.fwd .= suptree.bwd.mom .+ fwd.mom
            compute_criterion(suptree.summed_mom.fwd, suptree.bwd.dham_dmom, fwd.dham_dmom)
        else
            suptree.summed_mom.fwd .= tree.summed_mom.bwd .+ tree.summed_mom.fwd
            (
                compute_criterion(suptree.summed_mom.fwd, suptree.bwd.dham_dmom, fwd.dham_dmom) && 
                compute_criterion(badd(tree.summed_mom.bwd, tree.bwd.mom), suptree.bwd.dham_dmom, tree.bwd.dham_dmom) &&
                compute_criterion(badd(tree.bwd_fwd.mom, tree.summed_mom.fwd), tree.bwd_fwd.dham_dmom, fwd.dham_dmom)
            )
        end
    end
    start_tree!(depth) = if depth == 1 
        stepfwd!(__self__)
        # dham could be computed reactively if the fwd.ham invalidation propagated
        dham = finiteorneginf(init.ham - fwd.ham)
        collectstats!(__self__)
        diverged && return may_continue = false
        trees[1].log_weight[1] = dham
        rcopy!(proposals[1], fwd)
    else 
        start_tree!(__self__, depth-1)
        may_continue || return may_sample = false
        swapproposal!(__self__, depth-1, depth)
        finish_tree!(__self__, depth-1)
        if may_sample && randbernoullilog(rng, logadvanceprob(__self__, depth))
            swapproposal!(__self__, depth-1, depth)
        end
    end
end