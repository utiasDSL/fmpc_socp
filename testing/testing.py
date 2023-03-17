import seaborn as sns
sns.set(style="whitegrid")
from quad_1D.quad_1d import Quad1D
from controllers.fmpc import FMPC
from quad_1D.expr_utils import feedback_loop
from quad_1D.controllers import SOCPProblem
import control

# Model Parameters
dt = 0.02 # Discretization of simulation
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
horizon = 50
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

Q = fmpc.Q
R = fmpc.R
Ad = quad.Ad
Bd = quad.Bd

K, P, E = control.dlqr(Ad, Bd, Q, R)
