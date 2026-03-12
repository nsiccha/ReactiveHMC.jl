tr_prod(A::AbstractMatrix, B::AbstractMatrix) = sum(Base.broadcasted(*, A', B))

@reactive euclidean_phasepoint(pot_f, grad_f, metric, pos, mom) = begin 
    pot = pot_f(pos)
    pot, dpot_dpos = grad_f(pos)

    chol_metric = cholesky(metric)
    dkin_dmom = chol_metric \ mom
    # The logdet term could be left out as it doesn't change (unless the metric changes)
    kin = .5 * (@node(logdet(chol_metric)) + dot(mom, dkin_dmom))

    ham = pot + kin
    dham_dpos = dpot_dpos
    dham_dmom = dkin_dmom
end

@reactive riemannian_phasepoint(pot_f, grad_f, metric_f, metric_grad_f, pos, mom) = begin
    pot = pot_f(pos)
    pot, dpot_dpos = grad_f(pos)
    pot, dpot_dpos, metric = metric_f(pos)
    pot, dpot_dpos, metric, metric_grad = metric_grad_f(pos)

    chol_metric = cholesky(metric)
    dkin_dmom = chol_metric \ mom

    kin = .5 * (@node(logdet(chol_metric)) + dot(mom, dkin_dmom))

    inv_metric = Symmetric(inv(chol_metric))
    dkin_dpos .= @node(map(eachslice(metric_grad; dims=3)) do pgi
        .5 * tr_prod(inv_metric, pgi)
    end) .- Base.broadcasted(eachslice(metric_grad; dims=3)) do pgi
        .5 * dot(dkin_dmom, pgi, dkin_dmom)
    end

    ham = pot + kin
    @. dham_dpos = dkin_dpos + dpot_dpos
    dham_dmom = dkin_dmom
end

@reactive riemannian_softabs_phasepoint(pot_f, grad_f, premetric_f, premetric_grad_f, pos, mom; alpha=20.) = begin 
    pot = pot_f(pos)
    pot, dpot_dpos = grad_f(pos)
    pot, dpot_dpos, premetric = premetric_f(pos)
    pot, dpot_dpos, premetric, premetric_grad = premetric_grad_f(pos)

    premetric_eigvals, Q = eigen(Symmetric(premetric))
    @. metric_eigvals = premetric_eigvals * coth(alpha * premetric_eigvals)
    @. Q_inv = Q / metric_eigvals'
    Qp_mom = Q' * mom
    Q_inv_Qp_mom = Q_inv * Diagonal(Qp_mom)

    dkin_dmom = Q_inv * Qp_mom
    kin = .5 * (@node(sum(log, metric_eigvals)) + dot(mom, dkin_dmom))
    J .= Base.broadcasted(premetric_eigvals, metric_eigvals, premetric_eigvals', metric_eigvals') do pei, ei, pej, ej
        if pei == pej
            coth(alpha * pei) + pei * alpha * -csch(pei * alpha)^2 
        else
            ((ei - ej) / (pei - pej)) 
        end
    end
    dkin_dpos .= @node(map(eachslice(premetric_grad; dims=3)) do pgi
        .5 * tr_prod(@node(@node(Q_inv * Diagonal(J)) * Q'), pgi)
    end) .- Base.broadcasted(eachslice(premetric_grad; dims=3)) do pgi
        # I wonder how clever Julia is about computing the below matrix product
        .5 * tr_prod(@node(@node(Q_inv_Qp_mom * J) * Q_inv_Qp_mom'), pgi)
    end

    ham = pot + kin
    @. dham_dpos = dkin_dpos + dpot_dpos
    dham_dmom = dkin_dmom
end

@reactive relativistic_euclidean_phasepoint(pot_f, grad_f, metric, pos, mom; c, m) = begin 
    pot = pot_f(pos)
    pot, dpot_dpos = grad_f(pos)

    chol_metric = cholesky(metric)
    dprekin_dmom = chol_metric \ mom
    kin_sqrt_term = m*sqrt(1+dot(mom, dprekin_dmom)/(m*c)^2)
    dkin_dmom .= kin_sqrt_term .\ dprekin_dmom
    kin = @node(.5 * logdet(chol_metric)) + c^2*kin_sqrt_term

    ham = pot + kin
    dham_dpos = dpot_dpos
    dham_dmom = dkin_dmom
end

@reactive relativistic_riemannian_phasepoint(pot_f, grad_f, metric_f, metric_grad_f, pos, mom; c, m) = begin 
    pot = pot_f(pos)
    pot, dpot_dpos = grad_f(pos)
    pot, dpot_dpos, metric = metric_f(pos)
    pot, dpot_dpos, metric, metric_grad = metric_grad_f(pos)

    chol_metric = cholesky(metric)
    dprekin_dmom = chol_metric \ mom
    kin_sqrt_term = m*sqrt(1+dot(mom, dprekin_dmom)/(m*c)^2)
    dkin_dmom .= kin_sqrt_term .\ dprekin_dmom
    kin = @node(.5 * logdet(chol_metric)) + c^2*kin_sqrt_term

    inv_metric = Symmetric(inv(chol_metric))
    dkin_dpos .= @node(map(eachslice(metric_grad; dims=3)) do pgi
        .5 * tr_prod(inv_metric, pgi)
    end) .- Base.broadcasted(eachslice(metric_grad; dims=3)) do pgi
        .5 * dot(dprekin_dmom, pgi, dkin_dmom)
    end

    ham = pot + kin
    @. dham_dpos = dkin_dpos + dpot_dpos
    dham_dmom = dkin_dmom
end

@reactive relativistic_riemannian_softabs_phasepoint(pot_f, grad_f, premetric_f, premetric_grad_f, pos, mom; alpha=20., c, m) = begin 
    pot = pot_f(pos)
    pot, dpot_dpos = grad_f(pos)
    pot, dpot_dpos, premetric = premetric_f(pos)
    pot, dpot_dpos, premetric, premetric_grad = premetric_grad_f(pos)

    premetric_eigvals, Q = eigen(Symmetric(premetric))
    @. metric_eigvals = premetric_eigvals * coth(alpha * premetric_eigvals)
    @. Q_inv = Q / metric_eigvals'
    D = Q' * mom

    dprekin_dmom = Q_inv * D
    kin_sqrt_term = m*sqrt(1+dot(mom, dprekin_dmom)/(m*c)^2)
    dkin_dmom .= kin_sqrt_term .\ dprekin_dmom
    kin = @node(.5 * sum(log, metric_eigvals)) + c^2*kin_sqrt_term

    J .= Base.broadcasted(premetric_eigvals, metric_eigvals, premetric_eigvals', metric_eigvals') do pei, ei, pej, ej
        if pei == pej
            coth(alpha * pei) + pei * alpha * -csch(pei * alpha)^2 
        else
            ((ei - ej) / (pei - pej)) 
        end
    end
    dkin_dpos .= @node(map(eachslice(premetric_grad; dims=3)) do pgi
        # log determinant contribution
        .5 * tr_prod(@node(Q_inv * Diagonal(J) * Q'), pgi)
    end) .- kin_sqrt_term .\ Base.broadcasted(eachslice(premetric_grad; dims=3)) do pgi
        # I wonder how clever Julia is about computing the below matrix product
        .5 * tr_prod(@node(Q_inv * (D .* J .* D') * Q_inv'), pgi)
    end

    ham = pot + kin
    @. dham_dpos = dkin_dpos + dpot_dpos
    dham_dmom = dkin_dmom
end