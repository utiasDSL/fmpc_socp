import seaborn as sns
sns.set(style="whitegrid")
from quad_1D.quad_1d import Quad1D
from controllers.dlqr import DLQR
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
ref_type = 'step'
# Define 2d quadrotor and reference traj
quad = Quad1D(T=Thrust, tau=tau, gamma=gamma, dt=dt, ref_type=ref_type)

T_prior = 7.0 # Thrust
tau_prior = 0.15 # Time constant
gamma_prior = 0 # Drag
quad_prior = Quad1D(T=T_prior, tau=tau_prior, gamma=gamma_prior, dt=dt, ref_type=ref_type)


# Controller Parameters
horizon = 100
q_lqr = [10.0, 0.0, 0.0]
r_lqr = [0.1]
#dlqr = DLQR(quad=quad_prior,
#            horizon=horizon,
#            dt=dt,
#            q_lqr=q_lqr,
#            r_lqr=r_lqr)

dlqr = DLQR(quad=quad,
            horizon=horizon,
            dt=dt,
            q_lqr=q_lqr,
            r_lqr=r_lqr)

# Reference
Amp = 0.1
omega = 5.0
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
    dlqr, # FB ctrl
    secondary_controllers=None, # No comparison
    online_learning=False,
    fig_count=0,
    plot=True
)

