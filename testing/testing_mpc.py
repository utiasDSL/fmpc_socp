import seaborn as sns
sns.set(style="whitegrid")
from quad_1D.quad_1d import Quad1D
from controllers.mpc import MPC
from quad_1D.expr_utils import feedback_loop

# Model Parameters
dt = 0.05 # Discretization of simulation
T = 10.0 # Simulation time
N = int(T/dt) # Number of time step
# Taken from LCSS 2021 paper
Thrust = 10 # Thrust
tau = 0.2 # Time constant
gamma = 3 # Drag
ref_type = 'increasing_sine'
# Define 2d quadrotor and reference traj
quad = Quad1D(thrust=Thrust, tau=tau, gamma=gamma, dt=dt, ref_type=ref_type)


T_prior = 10 # Thrust
tau_prior = 0.2 # Time constant
gamma_prior = 3 # Drag
quad_prior = Quad1D(thrust=T_prior, tau=tau_prior, gamma=gamma_prior, dt=dt, ref_type=ref_type)

# Controller Parameters
horizon = 20
q_mpc = [10.0, 1.0, 1.0]
r_mpc = [0.1]
solver = 'ipopt'
mpc = MPC(#quad=quad_prior,
          quad=quad,
          horizon=horizon,
          dt=dt,
          q_mpc=q_mpc,
          r_mpc=r_mpc,
          solver=solver)
mpc.reset()


# Reference
Amp = 0.2
omega = 0.9
reference_generator = quad.reference_generator
# Probabilistic guarantee of 1-delta
delta = 0.05
beta = 2.0
# Robust lqr smoothness term near error of zero
eps = 0.0001
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
    mpc, # FB ctrl
    secondary_controllers=None, # No comparison
    online_learning=False,
    fig_count=0,
    plot=True
)

#x_from_z = quad.cs_x_fom_z(z=fmpc_data_i['z'].T)['x'].toarray()
#
#x_int = np.zeros_like(fmpc_data_i['z'].T)
#x_int[:,0] = x_from_z[:,0]
#us = fmpc_data_i['u'][:-1]
#N = us.shape[0]
#for i in range(N):
#    x_int[:, i+1] = quad.cs_nonlin_dyn_discrete(x0=x_int[:,i],p=us[i])['xf'].toarray().squeeze()
#plt.figure()
#plt.plot(x_int[0,:], label='From Nonliner dyn')
#plt.plot(fmpc_data_i['z'][:,0], label='From Z')
#plt.title('Comparing different integration methods')
#plt.legend()
#plt.show()
