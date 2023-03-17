from sympy import *
import numpy as np
from quad_1D.quad_1d import Quad1D
import matplotlib.pyplot as plt
init_printing(use_unicode=True)
t, delta_t, omega, amp = symbols('t delta_t omega amp')
z_0 = amp*(0.5 + 0.5*tanh((t-delta_t)*omega))
z_1 = diff(z_0, t)
z_2 = diff(z_1, t)
v_ref = diff(z_2,t)

pprint('z_0:')
pprint(z_0)
pprint('z_1:')
pprint(z_1)
pprint('z_2:')
pprint(z_2)
pprint('vref:')
pprint(v_ref)

dt = 0.01
T_prior = 7 # Thrust
tau_prior = 0.15 # Time constant
gamma_prior = 0.0 # Drag
quad_prior = Quad1D(T=T_prior, tau=tau_prior, gamma=gamma_prior, dt=dt, ref_type='increasing_freq')
Amp = 0.2
omega = 20.0
t = np.arange(0, 5, dt)

z_ref, v_ref = quad_prior.reference_generator(t, Amp, omega, ref_type='step')
