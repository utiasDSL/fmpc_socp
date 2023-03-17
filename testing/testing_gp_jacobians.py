import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from quad_1D.quad_1d import Quad1D
from quad_1D.expr_utils import feedback_loop
from quad_1D.controllers import LQR
from learning.gp_utils import ZeroMeanAffineGP, GaussianProcess, train_gp

# Model Parameters
dt = 0.01 # Discretization of simulation
T = 5.0 # Simulation time
N = int(T/dt) # Number of time step
# Taken from LCSS 2021 paper
T = 10 # Thrust
tau = 0.2 # Time constant
gamma = 3 # Drag
# Define 2d quadrotor and reference traj
quad = Quad1D(T=T, tau=tau, gamma=gamma, dt=dt)
reference_generator = quad.reference_generator
# Prior model
T_prior = 10 # Thrust
tau_prior = 0.15 # Time constant
gamma_prior = 0 # Drag
quad_prior = Quad1D(T=T_prior, tau=tau_prior, gamma=gamma_prior, dt=dt)
# Input bounds
theta_lim = 35/180*np.pi
# LQR matrices
Q = np.diag([10.0, 10.0, 10.0])
R = 0.1 * np.eye(quad.m)
# Reference
Amp = 0.4
omega = 1.0
# Probabilistic guarantee of 1-delta
delta = 0.05
prob_theshold = np.sqrt(1.0-delta)
beta = 2.0
# Robust lqr smoothness term near error of zero
eps = 0.0001
# Compute the LQR Gain matrix and ARE soln for the quad
quad.lqr_gain_and_ARE_soln(Q, R)
quad_prior.lqr_gain_and_ARE_soln(Q, R)
# simulation parameters
params = {}
params['N'] = N
params['n'] = 3
params['m'] = 1
params['dt'] = dt
params['Amp'] = Amp
params['omega'] = omega
# Gather Training Data for GP
print("Generating Training Data...")
proir_lqr_controller = LQR('Prior LQR', quad_prior, Q, R)
true_lqr_controller = LQR('True LQR', quad, Q, R)
prior_model_fb_data, fig_count = feedback_loop(
                                                params, # paramters
                                                None, # GP model
                                                quad.true_flat_dynamics, # flat dynamics to step with
                                                reference_generator, # reference
                                                proir_lqr_controller, # FB ctrl
                                                [true_lqr_controller] # comparison
                                               )
## Train the nonlinear inverse GP (z,u) -> v
interval = 1
z_train = prior_model_fb_data['z']
u_train = prior_model_fb_data['u']
v_measured_train = prior_model_fb_data['v']
train_input_inv = np.hstack((z_train[::interval,:], u_train[::interval]))
train_targets_inv = (v_measured_train[::interval] # use v_prior_log, u_prior_log, and z_prior_log
                 - quad_prior.v_from_u(u_train[::interval].T, z_train[::interval,:].T).T
                 + np.random.normal(0, 1.0, size=(int(N/interval),1)) )
# Train the gp
print("Training the GP (u,z) -> v..")
gp_type = ZeroMeanAffineGP
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp_inv = GaussianProcess(gp_type, likelihood, 1)
train_x_inv = torch.from_numpy(train_input_inv).double()
train_y_inv = torch.from_numpy(train_targets_inv).squeeze().double()
gp_inv.train(train_x_inv, train_y_inv, n_train=50)
fig_count = gp_inv.plot_trained_gp(prior_model_fb_data['t'][::interval,0], fig_count=fig_count)
# Train the nonlinear term GP (z,v) -> u
train_input_nl = np.hstack((z_train[::interval,:], v_measured_train[::interval]))
# output data = v_cmd (v_des) - v_measured) + nois to match 2021 Paper
v_cmd = prior_model_fb_data['v_des']
train_targets_nl = (v_cmd[::interval]   #
                  - v_measured_train[::interval] #quad_prior.u_from_v(v_measured_train[::interval].T, z_train[::interval,:].T).T
                  + np.random.normal(0, 2.0, size=(int(N/interval),1)) )
# Train the gp
print("Training the GP (v,z) -> u..")
gp_nl = GaussianProcess(gp_type, likelihood, 1)
train_x_nl = torch.from_numpy(train_input_nl).double()
train_y_nl = torch.from_numpy(train_targets_nl).squeeze().double()
gp_nl.train(train_x_nl, train_y_nl, n_train=50)
fig_count = gp_nl.plot_trained_gp(prior_model_fb_data['t'][::interval,0], fig_count=fig_count)
noise = 2.0
variance = 20.0
gpy_nl = train_gp(noise, variance, 4, train_input_nl, train_targets_nl)
means_gpy, covs_gpy = gpy_nl.predict(train_input_nl)
lower = means_gpy - np.sqrt(covs_gpy)*2
upper = means_gpy + np.sqrt(covs_gpy)*2
fig_count += 1
plt.figure(fig_count)
plt.fill_between(prior_model_fb_data['t'][::interval,0], lower[:,0], upper[:,0])
plt.plot(prior_model_fb_data['t'][::interval,0], means_gpy[:,0], 'r')
plt.plot(prior_model_fb_data['t'][::interval,0], train_targets_nl[:,0], 'k*')
plt.title('GPy GP')
plt.show()
