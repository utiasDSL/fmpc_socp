import os
import munch
import numpy as np
import torch
import gpytorch
from copy import deepcopy
from functools import partial

from utils.dir_utils import set_dir_from_config
from experiements.experiments import train_gp_v_from_u, train_gpmpc_LHS, Experiment
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
           'input_bound': 10.0/180.0*np.pi,
           #'state_bound': {'h': np.array([[1.0, 0, 0]]).T, 'b': 0.51, 'phi_p': 5.0}
           'state_bound': None
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
                      'output_dir': '../models/gp_v_from_u'
                      }
fmpc_config = {'horizon': 100, #int(1/config['dt']),
              'q_mpc': [50.0, 0.1, 0.1],
              'r_mpc': [0.1],
              'solver': 'ipopt',
               #'lower_bounds': {'z0': -10.0},
               #'upper_bounds': {'z0': config['state_bound']['b']}
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
               'Amp': 0.5,
               'omega': 0.9,
               'ref_type': 'step'
}
config['quad'] = quad_config
config['quad_prior'] = quad_prior_config
config['gp_v_from_u'] = gp_v_from_u_config
config['fmpc'] = fmpc_config
config['socp'] = socp_config
config['dlqr'] = dlrq_config
config['test_params'] = test_params
config = munch.munchify(config)
set_dir_from_config(config)
quad = Quad1D(**quad_config)
quad_prior = Quad1D(**quad_prior_config)


ref_type = config.test_params.ref_type
ref_gen = partial(quad.reference_generator, ref_type=ref_type)


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
                               state_bound=config.state_bound,
                               ctrl=deepcopy(fmpc),
                               gp=gp_inv)

# Controller Parameters

dlqr = DLQR(quad=quad,
            dt=config.dt,
            q_lqr=config.dlqr.q_lqr,
            r_lqr=config.dlqr.r_lqr,
            reference_generator=ref_gen)
dlqr_socp = DiscreteSOCPFilter('DLQR+SOCP',
                               quad_prior,
                               config.socp.beta,
                               d_weight=config.socp.d_weight,
                               input_bound=config.input_bound,
                               state_bound=config.state_bound,
                               ctrl=deepcopy(dlqr),
                               gp=gp_inv)

#exp = Experiment('mpc', quad, [dlqr_socp, fmpc_socp], ref_gen, test_params, config)
exp = Experiment('mpc', quad, [dlqr, fmpc_socp], ref_gen, test_params, config)
exp.run_experiment()
exp.plot_tracking()
exp.plot_tracking(plot_dims=[0], name='position')
exp.summarize_timings()
exp.plot_rmse()
