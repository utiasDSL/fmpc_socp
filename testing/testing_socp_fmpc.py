import seaborn as sns
import numpy as np
from copy import deepcopy
import shelve
sns.set(style="whitegrid")
import gpytorch
from quad_1D.quad_1d import Quad1D
from controllers.fmpc import FMPC
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
ref_type='increasing_sine'
# Define 2d quadrotor and reference traj
quad = Quad1D(thrust=Thrust, tau=tau, gamma=gamma, dt=dt, ref_type=ref_type)

T_prior = 7.0 # Thrust
tau_prior = 0.15 # Time constant
gamma_prior = 0 # Drag
quad_prior = Quad1D(thrust=T_prior, tau=tau_prior, gamma=gamma_prior, dt=dt, ref_type=ref_type)

# Controller Parameters
horizon = 100
q_mpc = [20.0, 15.0, 5.0]
#q_mpc = [50.0, 0.1, 0.1]
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


#dlqr = DLQR(quad=quad,
#            horizon=horizon,
#            dt=dt,
#            q_lqr=q_lqr,
#            r_lqr=r_lqr)

# Reference
Amp = 0.2
omega = 0.6
reference_generator = quad.reference_generator
# Probabilistic guarantee of 1-delta
delta = 0.05
beta = 2.0
d_weight=0.0 # lower weight makes less chattering.
input_bound = 45.0*np.pi/180.0

output_dir = '/home/ahall/Documents/UofT/code/dsl__projects__flatness_safety_filter/testing/results/affine_gp/saved/seed42_Mar-01-17-52-44_9bf1cd2'

gp_type = ZeroMeanAffineGP
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp_inv = GaussianProcess(gp_type, likelihood, 1, output_dir)
gp_inv.init_with_hyperparam(output_dir)

# SOCP Prob
fmpc_prob = DiscreteSOCPFilter('SOCP',
                               quad_prior,
                               beta,
                               d_weight=d_weight,
                               input_bound=input_bound,
                               state_bound=None,
                               ctrl=deepcopy(fmpc),
                               gp=gp_inv)

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
    None, #gp_inv, # GP model
    quad.true_flat_dynamics, # flat dynamics to step with
    reference_generator, # reference
    fmpc_prob, # FB ctrl
    secondary_controllers=None, # No comparison
    online_learning=False,
    fig_count=0,
    plot=True
)

