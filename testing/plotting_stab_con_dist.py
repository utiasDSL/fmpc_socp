import seaborn as sns
import numpy as np
sns.set(style="whitegrid")
import matplotlib.pyplot as plt

from quad_1D.quad_1d import Quad1D
from controllers.fmpc import FMPC
from quad_1D.expr_utils import feedback_loop
from quad_1D.controllers import SOCPProblem

# Model Parameters
dt = 0.01 # Discretization of simulation
T = 10.0 # Simulation time
N = int(T/dt) # Number of time step
# Taken from LCSS 2021 paper
Thrust = 10 # Thrust
tau = 0.2 # Time constant
gamma = 3 # Drag
# Define 2d quadrotor and reference traj
quad = Quad1D(T=Thrust, tau=tau, gamma=gamma, dt=dt)

T_prior = 7.0 # Thrust
tau_prior = 0.15 # Time constant
gamma_prior = 0 # Drag
quad_prior = Quad1D(T=T_prior, tau=tau_prior, gamma=gamma_prior, dt=dt)

# Controller Parameters
horizon = 100
q_mpc = [20.0, 15.0, 5.0]
r_mpc = [0.1]
solver = 'qrqp'
fmpc = FMPC(quad=quad_prior,
            horizon=horizon,
            dt=dt,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            solver=solver)
fmpc.reset()

# Controller Parameters
horizon = 100
q_mpc = [20.0, 15.0, 5.0]
r_mpc = [0.1]
solver = 'ipopt'
#upper_bounds = {'z0': 0.25}
upper_bounds = None
#lower_bounds = {'z0': -10}
lower_bounds = None
con_tol = 0.0
h = np.array([[1.0, 0.0, 0.0]]).T
bcon = 0.25
phi_p = 3.0
state_bounds = {'h': h,
                'b': bcon,
                'phi_p': phi_p}
fmpc = FMPC(quad=quad_prior,
            horizon=horizon,
            dt=dt,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            solver=solver,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            con_tol=0.0)
fmpc.reset()
# Reference
Amp = 0.5
omega = 1.0
reference_generator = quad.reference_generator
# Probabilistic guarantee of 1-delta
delta = 0.05
beta = 2.0

# simulation parameters
params = {}
params['N'] = N
params['n'] = quad.n
params['m'] = quad.m
params['dt'] = dt
params['Amp'] = Amp
params['omega'] = omega
fmpc_data_i, fig_count = feedback_loop(
    params, # paramters
    None, # GP model
    quad.true_flat_dynamics, # flat dynamics to step with
    reference_generator, # reference
    fmpc, # FB ctrl
    secondary_controllers=None, # No comparison
    online_learning=False,
    fig_count=0,
    plot=True
)
z =  fmpc_data_i['z'][:-1,:]
e = fmpc_data_i['z'][:-1,:] - fmpc_data_i['z_ref'][:-1,:]
psi = fmpc_data_i['v'] - fmpc_data_i['v_des']

u_max = 45.0/180.0*np.pi

Nw1 = (fmpc.Ad - fmpc.Bd @ fmpc.K).T @ fmpc.P @ fmpc.Bd
term_1 = -e @ Nw1 * psi
w2 = fmpc.Bd.T @ fmpc.P @ fmpc.Bd
term_2 = w2 * psi**2

v_pmax = quad_prior.cs_v_from_u(z=z.T, u=u_max)['v'].toarray() - fmpc_data_i['v_des'].T
v_nmax = quad_prior.cs_v_from_u(z=z.T, u=-u_max)['v'].toarray() - fmpc_data_i['v_des'].T
term_2_max = w2*np.maximum(np.abs(v_pmax) , np.abs(v_nmax)).squeeze()**2
all_term = term_1 + term_2

plt.figure()
plt.plot(np.abs(term_1), label="|| Term 1 ||")
plt.plot(np.abs(term_2), label="|| Term 2 ||")
plt.plot(np.abs(term_2_max).T, label="|| Term 2 max||")
plt.legend()
plt.show()
