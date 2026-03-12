using LambertW
@reactive rke(;m=1., c=1.) = begin 
    c1 = m*c^2
    c2 = (m*c)^2
    e_sq(x_sq) = c1*sqrt(x_sq/c2+1)
    p_sq(x_sq) = exp(-e_sq(__self__, x_sq))
    P_sq(x_sq) = -(2 * c2 * exp(-c1 *sqrt((c2 + x_sq)/c2)) * (1 + c1 * sqrt((c2 + x_sq)/c2)))/c1^2
    P0_sq = -(2 * c2 * exp(-c1) * (1 + c1))/c1^2
    cdf_sq(x_sq) = (P0_sq - P_sq(__self__, x_sq)) / P0_sq
    quantile_sq(q) = (c2 - c1^2 *c2 + 2* c2* lambertw(-(c1^2* P0_sq* (-1 + q))/(2* c2* exp(1)), -1) + c2* lambertw(-(c1^2 *P0_sq* (-1 + q))/(2 *c2 *exp(1)), -1)^2)/c1^2
end