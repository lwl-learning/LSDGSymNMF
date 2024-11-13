function lambda = getLambda(d2, v2, dv, dEd, dEv)
p1 = d2 ^ 2;
p2 = 3 * dv * d2;
p3 = 2 * dv ^ 2 + d2 * v2 - dEd;
p4 = dv * v2 - dEv;

% smallest positive root
r = roots([p1, p2, p3, p4]);
r = real(r(abs(imag(r)) <= 1e-5));
lambda = min(min(r(r > 0)), 1);
end