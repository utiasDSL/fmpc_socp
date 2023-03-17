from sympy import *
init_printing(use_unicode=True)
w1, w2, w3, s, mu, v, sigma = symbols('w1 w2 w3 s mu v sigma')
gamma1, gamma2, gamma3, gamma4, gamma5, u = symbols('gamma1 gamma2 gamma3 gamma4 gamma5 u')

d = w2*v**2 + w1*v - w3
a = -2*v*w2 - w1
A = w2
c = sqrt(sigma)*(a/2 + A*mu)
lam = sigma

M = exp(s*(d + a*mu + A*mu**2) + s**2*c**2/(1-2*s*lam))*(1-2*s*lam)**(-1/2)

dmds = diff(M,s)
print("Mean:")
pprint(dmds.subs(s,0))
print()
d2mds2 = diff(dmds,s)
print("Var:")
pprint(d2mds2.subs(s,0))

print('With gammas:')
mu_val = gamma1 + gamma2 * u
sigma_val = sqrt(gamma3 + gamma4*u + gamma5*u**2)
dmds = diff(M,s)
print("Mean:")
pprint(dmds.subs({s: 0, mu: mu_val, sigma: sigma_val}))
print()
d2mds2 = diff(dmds,s)
print("Var:")
pprint(d2mds2.subs({s: 0, mu: mu_val, sigma: sigma_val}))
