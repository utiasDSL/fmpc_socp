import munch
import seaborn as sns
import numpy as np
import gpytorch
import torch
sns.set(style="whitegrid")
from copy import deepcopy
from quad_1D.quad_1d import Quad1D
from controllers.mpc import MPC
from quad_1D.expr_utils import feedback_loop
from quad_1D.controllers import LQR
from learning.gpmpc_gp_utils import DataHandler, GaussianProcessCollection, ZeroMeanIndependentGPModel, combine_prior_and_gp
from utils.dir_utils import set_dir_from_config

config = { 'seed': 42,
           'output_dir': './results/',
           'tag': 'gpmpc_testing'}
config = munch.munchify(config)
set_dir_from_config(config)


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


T_prior = 7 # Thrust
tau_prior = 0.10 # Time constant
gamma_prior = 0.0 # Drag
quad_prior = Quad1D(thrust=T_prior, tau=tau_prior, gamma=gamma_prior, dt=dt, ref_type=ref_type)

run_gpmpc = True
# GP params
sigmas = 0.0001
noise = {'mean': [0.0, 0.0, 0.0],
         'std': [sigmas, sigmas, sigmas]}
num_samples = 1000
n_train = [2000, 2000, 2000]
lr = [0.05, 0.05, 0.05]
#noise = None


# Controller Parameters
horizon = 50
q_mpc = [10.0, 0.1, 0.1]
r_mpc = [0.1]
solver = 'ipopt'
mpc = MPC(quad=quad_prior,
          horizon=horizon,
          dt=dt,
          q_mpc=q_mpc,
          r_mpc=r_mpc,
          solver=solver)
mpc.reset()

prior_lqr_controller = LQR('Prior LQR', quad_prior, deepcopy(mpc.Q), deepcopy(mpc.R))

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
x_data = []
u_data = []
omega_list = [0.3, 0.5, 0.7, 0.9]
params = {}
params['N'] = N
params['n'] = quad.n
params['m'] = quad.m
params['dt'] = dt
params['Amp'] = Amp
for omega in omega_list:
    params['omega'] = omega
    fmpc_data_i, fig_count = feedback_loop(
        params, # paramters
        None, # GP model
        quad.true_flat_dynamics, # flat dynamics to step with
        reference_generator, # reference
        #prior_lqr_controller, # FB ctrl
        mpc, # FB ctrl
        secondary_controllers=None, # No comparison
        online_learning=False,
        fig_count=0,
        plot=False
    )

    x_data.append(quad.cs_x_from_z(z=fmpc_data_i['z'].T)['x'].toarray().T)
    u_data.append(fmpc_data_i['u'])
    mpc.reset()
prior_model = deepcopy(quad_prior.cs_lin_dyn)
save_dir = config.output_dir

dh = DataHandler(x_data=x_data,
                 u_data=u_data,
                 prior_model=prior_model,
                 save_dir=save_dir,
                 noise=noise,
                 num_samples=num_samples)

likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    constraint=gpytorch.constraints.GreaterThan(1e-6),
).double()
input_mask = [1,2,3]
target_mask = [1,2]
gp = GaussianProcessCollection(ZeroMeanIndependentGPModel,
                               likelihood,
                               len(target_mask),
                               input_mask=input_mask,
                               target_mask=target_mask,
                                                     )

gp.train(torch.from_numpy(dh.data.train_inputs),
         torch.from_numpy(dh.data.train_targets),
         torch.from_numpy(dh.data.test_inputs),
         torch.from_numpy(dh.data.test_targets),
         n_train=n_train,
         learning_rate=lr,
         gpu=True,
         dir=config.output_dir)

# Load a GP with fewer kernel points
gp_small = GaussianProcessCollection(ZeroMeanIndependentGPModel,
                               likelihood,
                               len(target_mask),
                               input_mask=input_mask,
                               target_mask=target_mask,
                                                     )
N_gp_small = 200
interval = int(np.ceil(N/N_gp_small))
gp_small.init_with_hyperparam(train_inputs=torch.from_numpy(dh.data.train_inputs[::interval,:]),
                              train_targets=torch.from_numpy(dh.data.train_targets[::interval,:]),
                              path_to_statedicts=config.output_dir)

gp_precict = gp_small.make_casadi_predict_func()
dyn_func = combine_prior_and_gp(prior_model, gp_precict, input_mask, target_mask)

x_next_pred = dyn_func(x0=dh.data.x_seq[0].T, p=dh.data.u_seq[0].T)['xf'].toarray()
x_next_pred_pr = prior_model(x0=dh.data.x_seq[0].T, p=dh.data.u_seq[0].T)['xf'].toarray()

pred_RMSE = np.sum((x_next_pred - dh.data.x_next_seq[0].T)**2)
prior_RMSE = np.sum((x_next_pred_pr - dh.data.x_next_seq[0].T)**2)
print(f'GP RMSE: {pred_RMSE}')
print(f'Prior RMSE: {prior_RMSE}')

#test traj
params['omega'] = 0.


mpc_prior_data, fig_count = feedback_loop(
    params, # paramters
    None, # GP model
    quad.true_flat_dynamics, # flat dynamics to step with
    reference_generator, # reference
    #prior_lqr_controller, # FB ctrl
    mpc, # FB ctrl
    secondary_controllers=None, # No comparison
    online_learning=False,
    fig_count=0,
    plot=True
)


if run_gpmpc:
    gpmpc = MPC(quad=quad_prior,
                horizon=horizon,
                dt=dt,
                q_mpc=q_mpc,
                r_mpc=r_mpc,
                solver=solver,
                dynamics=dyn_func)
    gpmpc.reset()
    gpmpc_data_i, fig_count = feedback_loop(
        params, # paramters
        None, # GP model
        quad.true_flat_dynamics, # flat dynamics to step with
        reference_generator, # reference
        #prior_lqr_controller, # FB ctrl
        gpmpc, # FB ctrl
        secondary_controllers=None, # No comparison
        online_learning=False,
        fig_count=0,
        plot=True
    )
