import os.path
import munch
import torch
import gpytorch
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from copy import deepcopy

from learning.gp_utils import ZeroMeanAffineGP, GaussianProcess
from learning.gpmpc_gp_utils import DataHandler, ZeroMeanIndependentGPModel, GaussianProcessCollection, get_LHS_samples, generate_samples_into_sequences
from utils.plotting_utils import scatter3d, plot_trained_gp
from quad_1D.expr_utils import feedback_loop
from utils.dir_utils import set_dir_from_config,mkdirs

class Experiment:
    def __init__(self, name, test_quad, ctrls, reference_generator, test_params, config):
            self.name = name
            self.test_quad = test_quad
            self.reference_generator = reference_generator
            self.test_params = test_params
            self.ctrls = ctrls
            config['params'] = test_params
            config['name'] = name

            self.config = config
            self.results_dict = None
            self.reset()

    def run_experiment(self, plot_run=False, fig_count=0):
        for ctrl in self.ctrls:
            data, fig_count = feedback_loop(
                self.test_params, # paramters
                None, # GP model
                self.test_quad.true_flat_dynamics, # flat dynamics to step with
                self.reference_generator, # reference
                ctrl, # FB ctrl
                secondary_controllers=None, # No comparison
                online_learning=False,
                fig_count=fig_count,
                plot=plot_run,
                input_bound=self.config.input_bound
            )
            self.results_dict[ctrl.name] = data
        save_name = os.path.join(self.config.output_dir, 'data.npz')
        np.savez(save_name, **self.results_dict)
        self.results_dict = munch.munchify(self.results_dict)

    def plot_tracking(self, plot_dims=[0,1,2], fig_count=0, name=None):
        # Plot the states along the trajectory and compare with reference.
        fig_count += 1
        n_plots = len(plot_dims)
        units = {0: 'm', 1: 'm/s', 2: 'm/s^2'}
        fig, ax = plt.subplots(n_plots, figsize=(10,10))
        if n_plots == 1:
            ax = [ax]
        for plt_id in plot_dims:
            for ctrl_name, ctrl_data in self.results_dict.items():
                if ctrl_data['infeasible']:
                    inf_ind = ctrl_data['infeasible_index']
                    ax[plt_id].plot(ctrl_data.t[:inf_ind,:], ctrl_data.z[:inf_ind, plt_id], label=ctrl_name)
                    ax[plt_id].plot(ctrl_data.t[inf_ind-1,:], ctrl_data.z[inf_ind-1, plt_id], 'rX')
                else:
                    ax[plt_id].plot(ctrl_data.t, ctrl_data.z[:, plt_id], label=ctrl_name)
            ax[plt_id].plot(ctrl_data.t, ctrl_data.z_ref[:, plt_id], '--k', label='ref')
            y_label = f'$z_{plt_id}\; (' + units[plt_id] + ')$'
            ax[plt_id].set_ylabel(y_label)
        ax[-1].set_xlabel('Time (s)')
        plt.legend()
        plt.tight_layout()
        if name is None:
            plt_name = os.path.join(self.config.output_dir, 'tracking_plot.eps')
        else:
            plt_name = os.path.join(self.config.output_dir, name+'.eps')
        plt.savefig(plt_name)
        plt.show()
        return fig_count

    def summarize_timings(self):
        headers = [['Algo', 'RMSE_all','RMSE x only','mean (s)', 'std (s)']]
        for ctrl_name, ctrl_data in self.results_dict.items():
            rmse = calc_rmse(ctrl_data['z'] - ctrl_data['z_ref'])
            rmse_x = calc_rmse_x(ctrl_data['z'] - ctrl_data['z_ref'])
            mean_t = np.mean(ctrl_data['solve_time'][1:])
            std_t = np.std(ctrl_data['solve_time'][1:])
            line = [ctrl_name, rmse, rmse_x, mean_t, std_t]
            headers.append(line)
        fname = os.path.join(self.config.output_dir,'solve_times.csv')
        with open(fname, 'w') as fopen:
            writer = csv.writer(fopen, delimiter=',')
            writer.writerows(headers)

    def plot_rmse(self):
        x_pos = list(range(len(self.ctrls)))
        rmses = []
        names = []
        for ctrl_name, ctrl_data in self.results_dict.items():
            rmse = calc_rmse(ctrl_data['z'] - ctrl_data['z_ref'])
            rmses.append(rmse)
            names.append(ctrl_name)
        fig, ax = plt.subplots()
        ax.bar(x_pos, rmses, width=1.0, tick_label=names)
        ax.set_ylabel('RMSE')
        fname = os.path.join(self.config.output_dir, 'rmse.eps')
        plt.savefig(fname)
        plt.show()


    def reset(self):
        self.results_dict = {}

def calc_rmse(e):
    return np.sqrt(np.mean(e**2))

def calc_rmse_x(e):
    return calc_rmse(e[:,0])

def train_gp_v_from_u(config, quad_prior, quad):
    # Gather training parameters
    config.gp_v_from_u.output_dir = os.path.join(config.output_dir,'gp_v_from_u')
    mkdirs(config.gp_v_from_u.output_dir)
    seed = config.seed
    dt = config.dt
    T = config.T
    Amp = config.gp_v_from_u.amp
    omegalist = config.gp_v_from_u.omegalist
    sig = config.gp_v_from_u.sig
    # Sampling data points
    N = config.gp_v_from_u.N
    n_train = config.gp_v_from_u.n_train
    lr = config.gp_v_from_u.lr

    # Collect Training Data
    inputs = []
    targets = []
    for omega in omegalist:
        t = np.arange(0,T, dt)
        z_ref, v_real = quad.reference_generator(t, Amp, omega)
        u_ref = quad.cs_u_from_v(z=z_ref, v=v_real)['u'].toarray()
        v_hat = quad_prior.cs_v_from_u(z=z_ref, u=u_ref)['v'].toarray()
        u_ref_prior = quad_prior.cs_u_from_v(z=z_ref, v=v_hat)['u'].toarray()

        noise = np.random.normal(0, sig, size=v_hat.shape)
        v_hat_noisy = v_real + noise
        inputs.append(torch.from_numpy(np.vstack((z_ref, u_ref))).double().T)
        targets.append(torch.from_numpy(v_real).double().T)
    inputs = torch.vstack(inputs)
    targets = torch.vstack(targets)


    interval = int(np.ceil(inputs.shape[0]/N))
    inputs = inputs[::interval, :]
    targets = targets[::interval, :]

    train_in, test_in, train_tar, test_tar  = train_test_split(inputs, targets, test_size=0.2, random_state=seed)

    # Setup GP
    gp_type = ZeroMeanAffineGP
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_inv = GaussianProcess(gp_type, likelihood, 1, config.gp_v_from_u.output_dir)

    fname = os.path.join(config.gp_v_from_u.output_dir, 'training_output.txt')
    orig_stdout = sys.stdout
    with open(fname,'w', 1) as print_to_file:
        sys.stdout = print_to_file
        gp_inv.train(train_in, train_tar.squeeze(), n_train=n_train, learning_rate=lr)
    sys.stdout = orig_stdout

    means, covs, preds = gp_inv.predict(test_in)
    errors = means - test_tar.squeeze()
    abs_errors = torch.abs(errors)
    rel_errors = abs_errors/torch.abs(test_tar.squeeze())

    #scatter3d(test_in[:,0], test_in[:,1], test_in[:,2], errors)

    # Show Quality on unseen trajectory
    Amp = 0.2
    omega = 0.6
    t = np.arange(0,10, dt)
    z_test, v_test_real = quad.reference_generator(t, Amp, omega)
    u_test = quad.cs_u_from_v(z=z_test, v=v_test_real)['u'].toarray()
    ref_gp_ins = torch.from_numpy(np.vstack((z_test, u_test))).T
    delv_pred, u_cov, preds = gp_inv.predict(ref_gp_ins)
    v_test_prior = quad_prior.cs_v_from_u(z=z_test, u=u_test)['v'].toarray()
    #v_pred = delv_pred.T + v_test_prior
    v_pred = delv_pred.T

    figcount = plot_trained_gp(v_test_real, v_pred, preds, fig_count=1, show=True)

    likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
    gp2 = GaussianProcess(gp_type, likelihood2, 1, config.gp_v_from_u.output_dir)
    gp2.init_with_hyperparam( config.gp_v_from_u.output_dir)

    delv_pred2, u_cov2, preds2 = gp2.predict(ref_gp_ins)
    v_pred2 = delv_pred2.T
    plot_trained_gp(v_test_real, v_pred2, preds2, fig_count=figcount, show=True)
    return gp2

def train_gpmpc_gp(config, quad, quad_prior, ctrl):

    config.gpmpc.gp_output_dir = os.path.join(config.output_dir,'gpmpc_gp')
    mkdirs(config.gpmpc.gp_output_dir)
    seed = config.seed
    fname = os.path.join(config.gpmpc.gp_output_dir, 'training_output.txt')

    dt = config.dt
    T = config.T
    N = int(T/dt)

    # GP params
    noise = config.gpmpc.noise
    num_samples = config.gpmpc.num_samples
    n_train = config.gpmpc.n_train
    lr = config.gpmpc.lr
    input_mask = config.gpmpc.input_mask
    target_mask = config.gpmpc.target_mask

    # Training Traj params
    Amp = config.gpmpc.amp
    omega_list =  config.gpmpc.omegalist
    ctrl.reset()

    reference_generator = quad.reference_generator

    # simulation parameters
    x_data = []
    u_data = []
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
            ctrl, # FB ctrl
            secondary_controllers=None, # No comparison
            online_learning=False,
            fig_count=0,
            plot=False
        )

        x_data.append(quad.cs_x_from_z(z=fmpc_data_i['z'].T)['x'].toarray().T)
        u_data.append(fmpc_data_i['u'])
        ctrl.reset()
    prior_model = deepcopy(quad_prior.cs_lin_dyn)
    save_dir = config.gpmpc.gp_output_dir

    dh = DataHandler(x_data=x_data,
                     u_data=u_data,
                     prior_model=prior_model,
                     save_dir=save_dir,
                     noise=noise,
                     num_samples=num_samples)
    dh.save(config.gpmpc.gp_output_dir)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        constraint=gpytorch.constraints.GreaterThan(1e-6),
    ).double()
    gp = GaussianProcessCollection(ZeroMeanIndependentGPModel,
                                   likelihood,
                                   len(target_mask),
                                   input_mask=input_mask,
                                   target_mask=target_mask,
                                   )

    orig_stdout = sys.stdout
    with open(fname,'w', 1) as print_to_file:
        sys.stdout = print_to_file
        gp.train(torch.from_numpy(dh.data.train_inputs),
                 torch.from_numpy(dh.data.train_targets),
                 torch.from_numpy(dh.data.test_inputs),
                 torch.from_numpy(dh.data.test_targets),
                 n_train=n_train,
                 learning_rate=lr,
                 gpu=True,
                 dir=save_dir)
    sys.stdout = orig_stdout

def train_gpmpc_LHS(config, quad, quad_prior, ctrl):
    """ Using Latin Hypercube Sampling to get training data."""
    config.gpmpc.gp_output_dir = os.path.join(config.output_dir,'gpmpc_gp')
    mkdirs(config.gpmpc.gp_output_dir)
    seed = config.seed
    fname = os.path.join(config.gpmpc.gp_output_dir, 'training_output.txt')

    dt = config.dt
    T = config.T
    N = int(T/dt)

    # GP params
    noise = config.gpmpc.noise
    num_samples = config.gpmpc.num_samples
    n_train = config.gpmpc.n_train
    lr = config.gpmpc.lr
    input_mask = config.gpmpc.input_mask
    target_mask = config.gpmpc.target_mask

    # Training Traj params
    Amp = config.gpmpc.amp
    omega_list =  config.gpmpc.omegalist
    ctrl.reset()

    reference_generator = quad.reference_generator

    prior_model = deepcopy(quad_prior.cs_lin_dyn)
    save_dir = config.gpmpc.gp_output_dir

    LHS_sampler_args = {'lower_bounds': config.lhs_samp.lower_bounds,
                        'upper_bounds': config.lhs_samp.upper_bounds,
                        'num_samples': num_samples,
                        'seed': seed}

    x_data, u_data = generate_samples_into_sequences(get_LHS_samples, LHS_sampler_args, quad)
    dh = DataHandler(x_data=x_data,
                     u_data=u_data,
                     prior_model=prior_model,
                     save_dir=save_dir,
                     noise=noise,
                     num_samples=num_samples)
    dh.save(config.gpmpc.gp_output_dir)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        constraint=gpytorch.constraints.GreaterThan(1e-6),
    ).double()
    gp = GaussianProcessCollection(ZeroMeanIndependentGPModel,
                                   likelihood,
                                   len(target_mask),
                                   input_mask=input_mask,
                                   target_mask=target_mask,
                                   )

    orig_stdout = sys.stdout
    with open(fname,'w', 1) as print_to_file:
        sys.stdout = print_to_file
        gp.train(torch.from_numpy(dh.data.train_inputs),
                 torch.from_numpy(dh.data.train_targets),
                 torch.from_numpy(dh.data.test_inputs),
                 torch.from_numpy(dh.data.test_targets),
                 n_train=n_train,
                 learning_rate=lr,
                 gpu=True,
                 dir=save_dir)
    sys.stdout = orig_stdout
