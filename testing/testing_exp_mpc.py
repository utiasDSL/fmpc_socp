import munch
from quad_1D.quad_1d import Quad1D
from controllers.mpc import MPC
from controllers.fmpc import FMPC
from experiments.experiments import Experiment

# Model Parameters
dt = 0.02 # Discretization of simulation
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
tau_prior = 0.10 # Time constant
gamma_prior = 0 # Drag
quad_prior = Quad1D(T=T_prior, tau=tau_prior, gamma=gamma_prior, dt=dt, ref_type=ref_type)

horizon = 50
q_mpc = [10.0, 1.0, 1.0]
r_mpc = [0.1]
solver = 'ipopt'
mpc = MPC(quad=quad_prior,
          horizon=horizon,
          dt=dt,
          q_mpc=q_mpc,
          r_mpc=r_mpc,
          solver=solver)
mpc.reset()

# Controller Parameters
horizon = 50
q_fmpc = [20.0, 15.0, 5.0]
#q_fmpc = [10.0,  0.0, 0.0]
#q_mpc = [1.0,  1.0, 1.0]
r_fmpc = [0.1]
solver = 'ipopt'
fmpc = FMPC(quad=quad_prior,
            horizon=horizon,
            dt=dt,
            q_mpc=q_fmpc,
            r_mpc=r_fmpc,
            solver=solver)
fmpc.reset()

reference_generator = quad.reference_generator

Amp = 0.2
omega = 0.9
config = { 'seed': 42,
           'output_dir': './results/',
           'tag': 'mpc_testing'}
config = munch.munchify(config)
params = {}
params['N'] = N
params['n'] = quad.n
params['m'] = quad.m
params['dt'] = dt
params['Amp'] = Amp
params['omega'] = omega
exp = Experiment('mpc', quad, [mpc, fmpc], reference_generator, params, config)
exp.run_experiment()
exp.plot_tracking()
