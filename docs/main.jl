using Chairmarks, ReactiveHMC, LinearAlgebra
import AdvancedHMC: SoftAbsRiemannianMetric, RiemannianMetric, GaussianKinetic, Hamiltonian, step, GeneralizedLeapfrog, PhasePoint, DualValue


bench(dim; stepsize=.1, n_fi_steps=10, n_steps=1) = begin
    pos = randn(dim)
    mom = randn(dim)

    display(@be step(
        GeneralizedLeapfrog(stepsize, n_fi_steps), 
        (Hamiltonian(
            SoftAbsRiemannianMetric((dim,), last ∘ premetric_f, last ∘ premetric_grad_f, 20.),
            GaussianKinetic(), 
            pot_f, grad_f
        )), 
        (PhasePoint(
            pos, mom, DualValue(grad_f(copy(pos))...), DualValue(grad_f(copy(pos))...)
        )), n_steps
    ))

    display(@be multistep(generalized_leapfrog!, 
        (riemannian_softabs_phasepoint(pot_f, grad_f, premetric_f, premetric_grad_f, pos, mom)); 
        stepsize, n_fi_steps, n_steps
    ))
end
begin

    pot_f(pos) = .5sum(abs2, pos)
    grad_f(pos) = (pot_f(pos), +pos)
    premetric_f(pos) = (grad_f(pos)..., Diagonal(ones(length(pos))))
    premetric_grad_f(pos) = (premetric_f(pos)..., zeros((length(pos), length(pos), length(pos))))
    struct Counter{F}
        n::Ref{Int}
        f::F
        Counter(f) = new{typeof(f)}(Ref(0), f)
    end
    (c::Counter)(args...; kwargs...) = begin
        c.n[] += 1
        c.f(args...; kwargs...)
    end
    # ENV["JULIA_DEBUG"] = ReactiveObjects
    dim = 1
    pos = randn(dim)
    mom = randn(dim)
    stepsize = .1
    n_fi_steps = 10
    n_steps = 20
    f1 = Counter(pot_f)
    f2 = Counter(grad_f)
    f3 = Counter(premetric_f)
    f4 = Counter(premetric_grad_f)
    obj = riemannian_softabs_phasepoint(f1, f2, f3, f4, pos, mom)
    multistep(generalized_leapfrog!, obj; stepsize, n_fi_steps, n_steps)
    display((f1.n[], f2.n[], f3.n[], f4.n[]))
    
    f1.n[] = f2.n[] = f3.n[] = f4.n[] = 0
    step(
        GeneralizedLeapfrog(stepsize, n_fi_steps), 
        (Hamiltonian(
            SoftAbsRiemannianMetric((dim,), last ∘ f3, last ∘ f4, 20.),
            GaussianKinetic(), 
            f1, f2
        )), 
        (PhasePoint(
            pos, mom, DualValue(grad_f(copy(pos))...), DualValue(grad_f(copy(pos))...)
        )), n_steps
    )
    display((f1.n[], f2.n[], f3.n[], f4.n[]))
end

begin 
    ENV["JULIA_DEBUG"] = nothing

    bench(1)
    for i in 4:7
        dim, n_fi_steps, n_steps = 2^i, 2^(10-i), 2^(10-i)
        @info (;dim, n_fi_steps, n_steps)
        bench(dim; n_fi_steps)
    end
end

begin
    using Plots, Random
    dim = 2
    p = plot()
    q = plot()
    permissive_leapfrog!(args...; n_fi_steps=missing, kwargs...) = leapfrog!(args...; kwargs...)
    for stepper in (
        # permissive_leapfrog!, 
        # implicit_midpoint!, 
        generalized_leapfrog!, 
        multistep(generalized_leapfrog!; n_steps=100)
    )
        rng = Xoshiro(0)
        pos = randn(rng, dim)
        mom = randn(rng, dim)
        c = m = 1.
        T = 160
        n_steps = 320
        stepsize = T / n_steps
        n_fi_steps = 10
        f1 = Counter(pot_f)
        f2 = Counter(grad_f)
        f3 = Counter(premetric_f)
        f4 = Counter(premetric_grad_f)
        # phasepoint = euclidean_phasepoint(f1, f2, last(f3(pos)), pos, mom)
        phasepoint = relativistic_euclidean_phasepoint(f1, f2, last(f3(pos)), pos, mom; c, m)
        # phasepoint = riemannian_softabs_phasepoint(f1, f2, f3, f4, pos, mom)
        # phasepoint = relativistic_riemannian_softabs_phasepoint(f1, f2, f3, f4, pos, mom; c, m)
        traj = zeros((dim, n_steps))
        hams = zeros(n_steps)
        @time for i in 1:n_steps
            # leapfrog!(phasepoint; stepsize)
            # implicit_midpoint!(phasepoint; stepsize, n_fi_steps)
            stepper(phasepoint; stepsize, n_fi_steps)
            traj[:, i] .= phasepoint.pos
            hams[i] = phasepoint.ham

        end
        display(stepper=>(f1.n[], f2.n[], f3.n[], f4.n[]))
    # plot(traj')
        plot!(p, eachrow(traj)...; marker=:circle, label=stepper)
        plot!(q, hams; marker=:circle, label=stepper)
    end
    display(plot(p, q; layout=(:, 1), size=(800, 800)))
    # obj = relativistic_riemannian_softabs_phasepoint(f1, f2, f3, f4, pos, mom; c, m)
    # multistep(generalized_leapfrog!, obj; stepsize, n_fi_steps, n_steps)
    # display((f1.n[], f2.n[], f3.n[], f4.n[]))

end