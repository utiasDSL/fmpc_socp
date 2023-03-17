import seaborn as sns
sns.set(style="whitegrid")
import numpy as np
import torch
import gpytorch
import munch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from quad_1D.quad_1d import Quad1D
from learning.gp_utils import ZeroMeanAffineGP, GaussianProcess, train_gp
from utils.plotting_utils import scatter3d, plot_trained_gp
from utils.dir_utils import set_dir_from_config

config = { 'seed': 42,
           'output_dir': './results/',
           'tag': 'affine_gp'}
config = munch.munchify(config)
set_dir_from_config(config)

seed = 42

# Model Parameters
dt = 0.01 # Discretization of simulation
T = 10.0 # Simulation time
N = int(T/dt) # Number of time step
# Taken from LCSS 2021 paper
Thrust = 10 # Thrust
tau = 0.2 # Time constant
gamma = 3 # Drag
# Define 2d quadrotor and reference traj
quad = Quad1D(thrust=Thrust, tau=tau, gamma=gamma, dt=dt)

T_prior = 7.0 # Thrust
tau_prior = 0.15 # Time constant
gamma_prior = 0 # Drag
quad_prior = Quad1D(thrust=T_prior, tau=tau_prior, gamma=gamma_prior, dt=dt)

# Reference
Amp = 0.2
inputs = []
targets = []
omegalist = [0.3, 0.5, 0.7, 0.9]
sig = 0.0001
for omega in omegalist:
    t = np.arange(0,T, dt)
    #z_ref, v_real = quad.reference_generator(t, Amp, omega, ref_type='increasing_freq')
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

# Sampling data points
N = 1000
d = 4
n_train=500
lr=0.1
interval = int(np.ceil(inputs.shape[0]/N))
inputs = inputs[::interval, :]
targets = targets[::interval, :]

train_in, test_in, train_tar, test_tar  = train_test_split(inputs, targets, test_size=0.2, random_state=seed)

gp_type = ZeroMeanAffineGP
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp_inv = GaussianProcess(gp_type, likelihood, 1, config.output_dir)
gp_inv.train(train_in, train_tar.squeeze(), n_train=n_train, learning_rate=lr)

means, covs, preds = gp_inv.predict(test_in)
errors = means - test_tar.squeeze()
abs_errors = torch.abs(errors)
rel_errors = abs_errors/torch.abs(test_tar.squeeze())

scatter3d(test_in[:,0], test_in[:,1], test_in[:,2], errors)

# Reference
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
gp2 = GaussianProcess(gp_type, likelihood2, 1, config.output_dir)
gp2.init_with_hyperparam(config.output_dir)

delv_pred2, u_cov2, preds2 = gp2.predict(ref_gp_ins)
v_pred2 = delv_pred2.T
plot_trained_gp(v_test_real, v_pred2, preds2, fig_count=figcount, show=True)
