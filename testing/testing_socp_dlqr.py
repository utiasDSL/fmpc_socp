import seaborn as sns
import numpy as np
import shelve
sns.set(style="whitegrid")
import gpytorch
from quad_1D.quad_1d import Quad1D
from controllers.dlqr import DLQR
from quad_1D.expr_utils import feedback_loop
from controllers.discrete_socp_filter import DiscreteSOCPFilter
#from quad_1D.gp_utils import ZeroMeanAffineGP, GaussianProcess, train_gp
from learning.gp_utils import ZeroMeanAffineGP, GaussianProcess, train_gp


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
q_lqr = [20.0, 15.0, 5.0]
r_lqr = [0.1]
dlqr = DLQR(quad=quad_prior,
            dt=dt,
            q_lqr=q_lqr,
            r_lqr=r_lqr)

#dlqr = DLQR(quad=quad,
#            horizon=horizon,
#            dt=dt,
#            q_lqr=q_lqr,
#            r_lqr=r_lqr)

# Reference
Amp = 0.2
omega = 0.8
reference_generator = quad.reference_generator
# Probabilistic guarantee of 1-delta
delta = 0.05
beta = 2.0
d_weight=10000.0 # lower weight makes less chattering.
input_bound = 45.0*np.pi/180.0

output_dir = '/home/ahall/Documents/UofT/code/dsl__projects__flatness_safety_filter/testing/results/affine_gp/saved/seed42_Mar-01-17-52-44_9bf1cd2'

gp_type = ZeroMeanAffineGP
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp_inv = GaussianProcess(gp_type, likelihood, 1, output_dir)
gp_inv.init_with_hyperparam(output_dir)

# SOCP Prob
h = np.array([[1.0, 0.0, 0.0]]).T
bcon = 0.25
phi_p = 3.0
state_bounds = {'h': h,
                'b': bcon,
                'phi_p': phi_p}
dlqr_prob = DiscreteSOCPFilter('SOCP', quad_prior, beta, d_weight=d_weight, input_bound=input_bound, state_bound=state_bounds, ctrl=dlqr)

# simulation parameters
params = {}
params['N'] = N
params['n'] = quad.n
params['m'] = quad.m
params['dt'] = dt
params['Amp'] = Amp
params['omega'] = omega
socp_data, fig_count = feedback_loop(
    params, # paramters
    gp_inv, # GP model
    quad.true_flat_dynamics, # flat dynamics to step with
    reference_generator, # reference
    dlqr_prob, # FB ctrl
    secondary_controllers=None, # No comparison
    online_learning=False,
    fig_count=0,
    plot=True
)

