import os
import munch
import numpy as np
import torch
import gpytorch
from copy import deepcopy
from functools import partial

from utils.dir_utils import set_dir_from_config
from experiments.experiments import train_gp_v_from_u, train_gpmpc_LHS, Experiment
from quad_1D.quad_1d import Quad1D
from controllers.mpc import MPC
from controllers.fmpc import FMPC
from controllers.dlqr import DLQR
from controllers.discrete_socp_filter import DiscreteSOCPFilter

from learning.gpmpc_gp_utils import  ZeroMeanIndependentGPModel, GaussianProcessCollection, DataHandler, combine_prior_and_gp
from learning.gp_utils import ZeroMeanAffineGP, GaussianProcess

config = { 'seed': 42,
           'output_dir': './results/',
           'tag': 'tracking_comp',
           'dt': 0.02,
           'T': 10.0,
           'T_test': 10.0,
           'input_bound': 15.0/180.0*np.pi,
           #'state_bound': {'h': [1, 0, 0], 'b': 0.25, 'phi_p': 3.0}
            }
quad_config = {'thrust': 10,
               'tau': 0.2,
               'gamma': 3.0,
               'dt': config['dt']}
quad_prior_config = {'thrust': 20,
                     'tau': 0.05,
                     'gamma': 0.0,
                     'dt': config['dt']}
gp_v_from_u_config = {'amp': 0.2,
                      'omegalist': [0.3, 0.5, 0.7, 0.9],
                      'sig': 0.0001,
                      'N': 1000,
                      'n_train': 500,
                      'lr': 0.1,
                      #'output_dir': None
                      'output_dir': '/home/ahall/Documents/UofT/code/dsl__projects__flatness_safety_filter/testing/results/tracking_comp/saved/seed42_Mar-09-20-47-05_fe35b85/gp_v_from_u'
                      }
sigmas = 0.0001
gpmpc_config = {
         'amp': 0.2,
         'omegalist': [0.3, 0.5, 0.7, 0.9],
         'sig': 0.0001,
         'num_samples': 5000,
         'n_train': [2000, 2000, 2000],
         'lr': [0.05, 0.05, 0.05],
         'noise':  {'mean': [0.0, 0.0, 0.0], 'std': [sigmas, sigmas, sigmas]},
         'mpc_prior': {'horizon': 50, #int(1/config['dt']),
                       'q_mpc': [10.0, 0.1, 0.1],
                       'r_mpc': [0.1],
                       'input_bound': config['input_bound'],
                       'solver': 'ipopt'
                       },
         'input_mask': [1,2,3],
         'target_mask': [1,2],
         'pred_kern_size': 200,
         'gp_output_dir': '/home/ahall/Documents/UofT/code/dsl__projects__flatness_safety_filter/testing/results/tracking_comp/saved/seed42_Mar-09-20-47-05_fe35b85/gpmpc_gp'
         }
lhs_config = { 'lower_bounds': [-0.01, -2.0, -1.5, -0.6],
               'upper_bounds': [0.01, 2.0, 1.5, 0.6]}
fmpc_config = {'horizon': 50, #int(1/config['dt']),
              'q_mpc': [10.0, 0.1, 0.1],
              'r_mpc': [0.1],
              'solver': 'ipopt'
}
socp_config = {'d_weight': 0,
               'beta': 2.0}
dlrq_config = {'q_lqr': fmpc_config['q_mpc'],
               'r_lqr': fmpc_config['r_mpc']}
test_params = {
               'N': int(config['T_test']/config['dt']),
               'n': 3,
               'm': 1,
               'dt': config['dt'],
               'Amp': 0.2,
               'omega': 0.6,
               'ref_type': 'increasing_sine'
}
config['quad'] = quad_config
config['quad_prior'] = quad_prior_config
config['gp_v_from_u'] = gp_v_from_u_config
config['gpmpc'] = gpmpc_config
config['fmpc'] = fmpc_config
config['socp'] = socp_config
config['dlqr'] = dlrq_config
config['test_params'] = test_params
config['lhs_samp'] = lhs_config
config = munch.munchify(config)
set_dir_from_config(config)
quad = Quad1D(**quad_config)
quad_prior = Quad1D(**quad_prior_config)


horizon = config.gpmpc.mpc_prior.horizon
q_mpc = config.gpmpc.mpc_prior.q_mpc
r_mpc = config.gpmpc.mpc_prior.r_mpc
solver = config.gpmpc.mpc_prior.solver
ref_type = config.test_params.ref_type
ref_gen = partial(quad.reference_generator, ref_type=ref_type)
input_bound = config.gpmpc.mpc_prior.input_bound

if config.gp_v_from_u.output_dir is None:
    gp_inv_new = train_gp_v_from_u(config, quad_prior, quad)
if config.gpmpc.gp_output_dir is None:
    #train_gpmpc_gp(config, quad, quad_prior, mpc)
    train_gpmpc_LHS(config, quad, quad_prior, mpc)

input_mask = config.gpmpc.input_mask
target_mask = config.gpmpc.target_mask

likelihood = gpytorch.likelihoods.GaussianLikelihood(
    constraint=gpytorch.constraints.GreaterThan(1e-6)).double()
prior_model = deepcopy(quad_prior.cs_lin_dyn)
gp_small = GaussianProcessCollection(ZeroMeanIndependentGPModel,
                                     likelihood,
                                     len(target_mask),
                                     input_mask=input_mask,
                                     target_mask=target_mask)
N_gp_small = config.gpmpc.pred_kern_size
interval = int(np.ceil(config.gpmpc.num_samples*0.8/N_gp_small))
dh = DataHandler.load(os.path.join(config.gpmpc.gp_output_dir, 'data_handler'))
gp_small.init_with_hyperparam(train_inputs=torch.from_numpy(dh.data.train_inputs[::interval,:]),
                              train_targets=torch.from_numpy(dh.data.train_targets[::interval,:]),
                              path_to_statedicts=config.gpmpc.gp_output_dir)
gp_precict = gp_small.make_casadi_predict_func()
dyn_func = combine_prior_and_gp(prior_model, gp_precict, input_mask, target_mask)
# For testing
gpmpc = MPC(quad=quad_prior,
            name='GPMPC',
            horizon=horizon,
            dt=config.dt,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            solver=solver,
            dynamics=dyn_func,
            reference_generator=ref_gen,
            input_bound=input_bound)
gpmpc.reset()

mpc = MPC(quad=quad_prior,
          name='MPC',
          horizon=horizon,
          dt=config.dt,
          q_mpc=q_mpc,
          r_mpc=r_mpc,
          solver=solver,
          dynamics=prior_model,
          reference_generator=ref_gen,
          input_bound=input_bound)
mpc.reset()
fmpc = FMPC(quad=quad_prior,
            dt=config.dt,
            **config.fmpc,
            reference_generator=ref_gen)
fmpc.reset()


gp_type = ZeroMeanAffineGP
likelihood_inv = gpytorch.likelihoods.GaussianLikelihood()
gp_inv = GaussianProcess(gp_type, likelihood_inv, 1, config.gp_v_from_u.output_dir)
gp_inv.init_with_hyperparam(config.gp_v_from_u.output_dir)
fmpc_socp = DiscreteSOCPFilter('FMPC+SOCP',
                               quad_prior,
                               config.socp.beta,
                               d_weight=config.socp.d_weight,
                               input_bound=config.input_bound,
                               state_bound=None,
                               ctrl=deepcopy(fmpc),
                               gp=gp_inv)

# Controller Parameters

dlqr = DLQR(quad=quad_prior,
            dt=config.dt,
            q_lqr=config.dlqr.q_lqr,
            r_lqr=config.dlqr.r_lqr)
dlqr_socp = DiscreteSOCPFilter('DLQR+SOCP',
                               quad_prior,
                               config.socp.beta,
                               d_weight=config.socp.d_weight,
                               input_bound=config.input_bound,
                               state_bound=None,
                               ctrl=deepcopy(dlqr),
                               gp=gp_inv)

exp = Experiment('mpc', quad, [mpc, fmpc, dlqr, fmpc_socp, gpmpc, dlqr_socp], ref_gen, test_params, config)
exp.run_experiment()
exp.plot_tracking()
exp.plot_tracking(plot_dims=[0], name='position')
exp.summarize_timings()
exp.plot_rmse()
