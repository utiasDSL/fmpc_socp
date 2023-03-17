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
from learning.gpmpc_gp_utils import DataHandler, GaussianProcessCollection, ZeroMeanIndependentGPModel, combine_prior_and_gp, generate_samples_into_sequences, get_LHS_samples, get_MVN_samples
from utils.dir_utils import set_dir_from_config

config = { 'seed': 42,
           'output_dir': './results/',
           'tag': 'gpmpc_testing'}
config = munch.munchify(config)
set_dir_from_config(config)


# Model Parameters
dt = 0.02 # Discretization of simulation
T = 10.0 # Simulation time
N = int(T/dt) # Number of time step
# Taken from LCSS 2021 paper
Thrust = 10 # Thrust
tau = 0.2 # Time constant
gamma = 3 # Drag
ref_type = 'increasing_sine'
# Define 2d quadrotor and reference traj
quad = Quad1D(thrust=Thrust, tau=tau, gamma=gamma, dt=dt, ref_type=ref_type)


T_prior = 20 # Thrust
tau_prior = 0.05 # Time constant
gamma_prior = 0.0 # Drag
quad_prior = Quad1D(thrust=T_prior, tau=tau_prior, gamma=gamma_prior, dt=dt, ref_type=ref_type)

run_gpmpc = True
# GP params
sigmas = 0.0001
noise = {'mean': [0.0, 0.0, 0.0],
         'std': [sigmas, sigmas, sigmas]}
num_samples = 5000
n_train = [2000, 2000, 2000]
lr = [0.05, 0.05, 0.05]
#noise = None

# 0.05 Hz using LHS
#gp_load_path = '/home/ahall/Documents/UofT/code/dsl__projects__flatness_safety_filter/testing/results/gpmpc_testing/seed42_Mar-09-15-46-39_fe35b85'
# 0.02 Hz using LHS
#gp_load_path = '/home/ahall/Documents/UofT/code/dsl__projects__flatness_safety_filter/testing/results/gpmpc_testing/seed42_Mar-09-17-16-35_fe35b85'
gp_load_path = None
seed = 42
# LHS params
lb = [-0.01, -2.0, -1.5, -0.6 ]
ub = [0.01, 2.0, 1.5, 0.6]
LHS_sampler_args = {'lower_bounds': lb, 'upper_bounds': ub, 'num_samples': num_samples, 'seed': seed}

x_data, u_data = generate_samples_into_sequences(get_LHS_samples, LHS_sampler_args, quad)

#means = [0, 0, 0, 0]
#cov = np.diag([0.1, 0.2, 0.2, 0.2])
#MVN_sampler_args = {'means': means, 'cov': cov, 'num_samples': num_samples, 'seed': seed }
#x_data, u_data = generate_samples_into_sequences(get_MVN_samples, MVN_sampler_args, quad)

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


input_mask = [1,2,3]
target_mask = [1,2]

prior_model = deepcopy(quad_prior.cs_lin_dyn)
save_dir = config.output_dir
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    constraint=gpytorch.constraints.GreaterThan(1e-6),
).double()

dh = DataHandler(x_data=x_data,
                 u_data=u_data,
                 prior_model=prior_model,
                 save_dir=save_dir,
                 noise=noise,
                 num_samples=num_samples)
if gp_load_path is None:

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



#test traj
Amp = 0.2
omega = 0.5
params = {}
params['N'] = N
params['n'] = quad.n
params['m'] = quad.m
params['dt'] = dt
params['Amp'] = Amp
params['omega'] = omega
reference_generator = quad.reference_generator
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
    plot=False
)


# Load a GP with fewer kernel points
gp_small = GaussianProcessCollection(ZeroMeanIndependentGPModel,
                                     likelihood,
                                     len(target_mask),
                                     input_mask=input_mask,
                                     target_mask=target_mask,
                                     )
N_gp_small = 200
#interval = int(np.ceil(mpc_prior_data['z'].shape[0]*0.8/N_gp_small))
#dh_small = DataHandler(x_data=mpc_prior_data['z'],
#                       u_data=mpc_prior_data['u'],
#                       prior_model=prior_model,
#                       save_dir=save_dir,
#                       noise=noise,
#                       num_samples=mpc_prior_data['z'].shape[0])
#in_data_small, tar_data_small = dh_small.select_subsamples_with_kmeans(N_gp_small, seed)

#gp_small.init_with_hyperparam(train_inputs=torch.from_numpy(in_data_small),
#                              train_targets=torch.from_numpy(tar_data_small),
#                              path_to_statedicts=gp_load_path)

#gp_small.init_with_hyperparam(train_inputs=torch.from_numpy(dh_small.data.train_inputs[::interval,:]),
#                              train_targets=torch.from_numpy(dh_small.data.train_targets[::interval,:]),
#                              path_to_statedicts=gp_load_path)
interval = int(np.ceil(dh.data.train_inputs.shape[0]/N_gp_small))
gp_small.init_with_hyperparam(train_inputs=torch.from_numpy(dh.data.train_inputs[::interval,:]),
                              train_targets=torch.from_numpy(dh.data.train_targets[::interval,:]),
                              path_to_statedicts=config.output_dir)
                              #path_to_statedicts=gp_load_path)
gp_precict = gp_small.make_casadi_predict_func()
dyn_func = combine_prior_and_gp(prior_model, gp_precict, input_mask, target_mask)

#x_next_pred = dyn_func(x0=dh.data.x_seq[0].T, p=dh.data.u_seq[0].T)['xf'].toarray()
#x_next_pred_pr = prior_model(x0=dh.data.x_seq[0].T, p=dh.data.u_seq[0].T)['xf'].toarray()
#
#pred_RMSE = np.sum((x_next_pred - dh.data.x_next_seq[0].T)**2)
#prior_RMSE = np.sum((x_next_pred_pr - dh.data.x_next_seq[0].T)**2)
#print(f'GP RMSE: {pred_RMSE}')
#print(f'Prior RMSE: {prior_RMSE}')


if run_gpmpc:
    gpmpc = MPC(quad=quad_prior,
                name='GPMPC',
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
        plot=False
    )
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3)
for i in range(3):
    ax[i].plot(mpc_prior_data['z_ref'][:,i], label='ref')
    ax[i].plot(gpmpc_data_i['z'][:,i], label='GPMPC')
    ax[i].plot(mpc_prior_data['z'][:,i], label='MPC')
plt.legend()
plt.show()
