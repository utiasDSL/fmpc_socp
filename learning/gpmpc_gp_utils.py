"""Utility functions for Gaussian Processes.

"""
import os.path
import numpy as np
import gpytorch
import torch
import matplotlib.pyplot as plt
import casadi as cs
import shelve
from copy import deepcopy
from math import ceil
from munch import munchify
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from scipy.stats import qmc

from utils.dir_utils import mkdirs

torch.manual_seed(0)


def covSEard(x, z, ell, sf2):
    """GP squared exponential kernel.

    This function is based on the 2018 GP-MPC library by Helge-André Langåker

    Args:
        x (np.array or casadi.MX/SX): First vector.
        z (np.array or casadi.MX/SX): Second vector.
        ell (np.array or casadi.MX/SX): Length scales.
        sf2 (float or casadi.MX/SX): output scale parameter.

    Returns:
        SE kernel (casadi.MX/SX): SE kernel.

    """
    dist = cs.sum1((x - z) ** 2 / ell ** 2)
    return sf2 * cs.SX.exp(-.5 * dist)


class ZeroMeanIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    """Multidimensional Gaussian Process model with zero mean function.

    Or constant mean and radial basis function kernel (SE).

    """

    def __init__(self, train_x, train_y, likelihood, nx):
        """Initialize a multidimensional Gaussian Process model with zero mean function.

        Args:
            train_x (torch.Tensor): input training data (input_dim X N samples).
            train_y (torch.Tensor): output training data (output dim x N samples).
            likelihood (gpytorch.likelihood): Likelihood function (gpytorch.likelihoods.MultitaskGaussianLikelihood).
            nx (int): dimension of the target output (output dim)

        """
        super().__init__(train_x, train_y, likelihood)
        self.n = nx
        # For Zero mean function.
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([self.n]))
        # For constant mean function.
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([self.n]),
                                                                                    ard_num_dims=train_x.shape[1]),
                                                         batch_shape=torch.Size([self.n]),
                                                         ard_num_dims=train_x.shape[1])

    def forward(self, x):
        """

        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(gpytorch.distributions.MultivariateNormal(
            mean_x,
            covar_x))


class ZeroMeanIndependentGPModel(gpytorch.models.ExactGP):
    """Single dimensional output Gaussian Process model with zero mean function.

    Or constant mean and radial basis function kernel (SE).

    """

    def __init__(self, train_x, train_y, likelihood):
        """Initialize a single dimensional Gaussian Process model with zero mean function.

        Args:
            train_x (torch.Tensor): input training data (input_dim X N samples).
            train_y (torch.Tensor): output training data (output dim x N samples).
            likelihood (gpytorch.likelihood): Likelihood function (gpytorch.likelihoods.GaussianLikelihood).

        """
        super().__init__(train_x, train_y, likelihood)
        # For Zero mean function.
        self.mean_module = gpytorch.means.ZeroMean()
        # For constant mean function.
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]),
                                                         ard_num_dims=train_x.shape[1])

    def forward(self, x):
        """

        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcessCollection:
    """Collection of GaussianProcesses for multioutput GPs.

    """

    def __init__(self, model_type, likelihood, target_dim, input_mask=None, target_mask=None, normalize=False):
        """Creates a single GaussianProcess for each output dimension.

        Args:
            model_type (gpytorch model class): Model class for the GP (ZeroMeanIndependentGPModel).
            likelihood (gpytorch.likelihood): likelihood function.
            target_dim (int): Dimension of the output (how many GPs to make).
            input_mask (list): Input dimensions to keep. If None, use all input dimensions.
            target_mask (list): Target dimensions to keep. If None, use all target dimensions.
            normalize (bool): If True, scale all data between -1 and 1.

        """
        self.gp_list = []
        self.model_type = model_type
        self.likelihood = likelihood
        self.optimizer = None
        self.model = None
        self.NORMALIZE = normalize
        self.input_mask = input_mask
        self.target_mask = target_mask
        for i in range(target_dim):
            self.gp_list.append(GaussianProcess(model_type,
                                                deepcopy(likelihood),
                                                input_mask=input_mask,
                                                normalize=normalize))

    def _init_properties(self, train_inputs, train_targets):
        """Initialize useful properties.

        Args:
            train_inputs, train_targets (torch.tensors): Input and target training data.

        """
        target_dimension = train_targets.shape[1]
        self.input_dimension = train_inputs.shape[1]
        self.output_dimension = target_dimension
        self.n_training_samples = train_inputs.shape[0]

    def init_with_hyperparam(self, train_inputs, train_targets, path_to_statedicts):
        """Load hyperparameters from a state_dict.

        Args:
            train_inputs, train_targets (torch.tensors): Input and target training data.
            path_to_statedicts (str): Path to where the state dicts are saved.

        """
        self._init_properties(train_inputs, train_targets)
        target_dimension = train_targets.shape[1]
        gp_K_plus_noise_list = []
        gp_K_plus_noise_inv_list = []
        for gp_ind, gp in enumerate(self.gp_list):
            path = os.path.join(path_to_statedicts, 'best_model_%s.pth' % self.target_mask[gp_ind])
            print("#########################################")
            print("#       Loading GP dimension %s         #" % self.target_mask[gp_ind])
            print("#########################################")
            print('Path: %s' % path)
            gp.init_with_hyperparam(train_inputs, train_targets[:, self.target_mask[gp_ind]], path)
            gp_K_plus_noise_list.append(gp.model.K_plus_noise.detach())
            gp_K_plus_noise_inv_list.append(gp.model.K_plus_noise_inv.detach())
            print('Loaded!')
        gp_K_plus_noise = torch.stack(gp_K_plus_noise_list)
        gp_K_plus_noise_inv = torch.stack(gp_K_plus_noise_inv_list)
        self.K_plus_noise = gp_K_plus_noise
        self.K_plus_noise_inv = gp_K_plus_noise_inv
        self.casadi_predict = self.make_casadi_predict_func()

    def get_hyperparameters(self, as_numpy=False):
        """Get the outputscale and lengthscale from the kernel matrices of the GPs.

        """
        lengthscale_list = []
        output_scale_list = []
        noise_list = []
        for gp in self.gp_list:
            lengthscale_list.append(gp.model.covar_module.base_kernel.lengthscale.detach())
            output_scale_list.append(gp.model.covar_module.outputscale.detach())
            noise_list.append(gp.model.likelihood.noise.detach())
        lengthscale = torch.cat(lengthscale_list)
        outputscale = torch.Tensor(output_scale_list)
        noise = torch.Tensor(noise_list)
        if as_numpy:
            return lengthscale.numpy(), outputscale.numpy(), noise.numpy(), self.K_plus_noise.detach().numpy()
        else:
            return lengthscale, outputscale, noise, self.K_plus_noise

    def train(self,
              train_x_raw,
              train_y_raw,
              test_x_raw,
              test_y_raw,
              n_train=[500],
              learning_rate=[0.01],
              gpu=False,
              dir='results'):
        """Train the GP using Train_x and Train_y.

        Args:
            train_x: Torch tensor (N samples [rows] by input dim [cols])
            train_y: Torch tensor (N samples [rows] by target dim [cols])

        """
        self._init_properties(train_x_raw, train_y_raw)
        self.model_paths = []
        mkdirs(dir)
        gp_K_plus_noise_inv_list = []
        gp_K_plus_noise_list = []
        for gp_ind, gp in enumerate(self.gp_list):
            lr = learning_rate[self.target_mask[gp_ind]]
            n_t = n_train[self.target_mask[gp_ind]]
            print("#########################################")
            print("#      Training GP dimension %s         #" % self.target_mask[gp_ind])
            print("#########################################")
            print("Train iterations: %s" % n_t)
            print("Learning Rate:: %s" % lr)
            gp.train(train_x_raw,
                     train_y_raw[:, self.target_mask[gp_ind]],
                     test_x_raw,
                     test_y_raw[:, self.target_mask[gp_ind]],
                     n_train=n_t,
                     learning_rate=lr,
                     gpu=gpu,
                     fname=os.path.join(dir, 'best_model_%s.pth' % self.target_mask[gp_ind]))
            self.model_paths.append(dir)
            gp_K_plus_noise_list.append(gp.model.K_plus_noise)
            gp_K_plus_noise_inv_list.append(gp.model.K_plus_noise_inv)
        gp_K_plus_noise = torch.stack(gp_K_plus_noise_list)
        gp_K_plus_noise_inv = torch.stack(gp_K_plus_noise_inv_list)
        self.K_plus_noise = gp_K_plus_noise
        self.K_plus_noise_inv = gp_K_plus_noise_inv
        self.casadi_predict = self.make_casadi_predict_func()

    def predict(self, x, requires_grad=False, return_pred=True):
        """

        Args:
            x : torch.Tensor (N_samples x input DIM).

        Return
            Predictions
                mean : torch.tensor (nx X N_samples).
                lower : torch.tensor (nx X N_samples).
                upper : torch.tensor (nx X N_samples).

        """
        means_list = []
        cov_list = []
        pred_list = []
        for gp in self.gp_list:
            if return_pred:
                mean, cov, pred = gp.predict(x, requires_grad=requires_grad, return_pred=return_pred)
                pred_list.append(pred)
            else:
                mean, cov = gp.predict(x, requires_grad=requires_grad, return_pred=return_pred)
            means_list.append(mean)
            cov_list.append(cov)
        means = torch.tensor(means_list)
        cov = torch.diag(torch.cat(cov_list).squeeze())
        if return_pred:
            return means, cov, pred_list
        else:
            return means, cov

    def make_casadi_predict_func(self):
        """
        Assume train_inputs and train_tergets are already
        """

        means_list = []
        Nz = len(self.input_mask)
        Ny = len(self.target_mask)
        z = cs.SX.sym('z1', Nz)
        y = cs.SX.zeros(Ny)
        for gp_ind, gp in enumerate(self.gp_list):
            y[gp_ind] = gp.casadi_predict(z=z)['mean']
        casadi_predict = cs.Function('pred', [z], [y], ['z'], ['mean'])
        return casadi_predict

    def prediction_jacobian(self, query):
        """Return Jacobian.

        """
        raise NotImplementedError

    def plot_trained_gp(self, inputs, targets, fig_count=0):
        """Plot the trained GP given the input and target data.

        """
        for gp_ind, gp in enumerate(self.gp_list):
            fig_count = gp.plot_trained_gp(inputs,
                                           targets[:, self.target_mask[gp_ind], None],
                                           self.target_mask[gp_ind],
                                           fig_count=fig_count)
            fig_count += 1

    def _kernel_list(self, x1, x2=None):
        """Evaluate the kernel given vectors x1 and x2.

        Args:
            x1 (torch.Tensor): First vector.
            x2 (torch.Tensor): Second vector.

        Returns:
            list of LazyTensor Kernels.

        """
        if x2 is None:
            x2 = x1
        # todo: Make normalization at the GPCollection level?
        # if self.NORMALIZE:
        #    x1 = torch.from_numpy(self.gp_list[0].scaler.transform(x1.numpy()))
        #    x2 = torch.from_numpy(self.gp_list[0].scaler.transform(x2.numpy()))
        k_list = []
        for gp in self.gp_list:
            k_list.append(gp.model.covar_module(x1, x2))
        return k_list

    def kernel(self, x1, x2=None):
        """Evaluate the kernel given vectors x1 and x2.

        Args:
            x1 (torch.Tensor): First vector.
            x2 (torch.Tensor): Second vector.

        Returns:
            Torch tensor of the non-lazy kernel matrices.

        """
        k_list = self._kernel_list(x1, x2)
        non_lazy_tensors = [k.evaluate() for k in k_list]
        return torch.stack(non_lazy_tensors)

    def kernel_inv(self, x1, x2=None):
        """Evaluate the inverse kernel given vectors x1 and x2.

        Only works for square kernel.

        Args:
            x1 (torch.Tensor): First vector.
            x2 (torch.Tensor): Second vector.

        Returns:
            Torch tensor of the non-lazy inverse kernel matrices.

        """
        if x2 is None:
            x2 = x1
        assert x1.shape == x2.shape, ValueError("x1 and x2 need to have the same shape.")
        k_list = self._kernel_list(x1, x2)
        num_of_points = x1.shape[0]
        # Efficient inversion is performed VIA inv_matmul on the laze tensor with Identity.
        non_lazy_tensors = [k.inv_matmul(torch.eye(num_of_points).double()) for k in k_list]
        return torch.stack(non_lazy_tensors)


class GaussianProcess:
    """Gaussian Process decorator for gpytorch.

    """

    def __init__(self, model_type, likelihood, input_mask=None, target_mask=None, normalize=False):
        """Initialize Gaussian Process.

        Args:
            model_type (gpytorch model class): Model class for the GP (ZeroMeanIndependentMultitaskGPModel).
            likelihood (gpytorch.likelihood): likelihood function.
            normalize (bool): If True, scale all data between -1 and 1. (prototype and not fully operational).

        """
        self.model_type = model_type
        self.likelihood = likelihood
        self.optimizer = None
        self.model = None
        self.NORMALIZE = normalize
        self.input_mask = input_mask
        self.target_mask = target_mask

    def _init_model(self, train_inputs, train_targets):
        """Init GP model from train inputs and train_targets.

        """
        if train_targets.ndim > 1:
            target_dimension = train_targets.shape[1]
        else:
            target_dimension = 1

        if self.NORMALIZE:
            # Define normalization scaler.
            self.scaler = preprocessing.StandardScaler().fit(train_inputs.numpy())
            train_inputs = torch.from_numpy(self.scaler.transform(train_inputs.numpy()))

        if self.model is None:
            self.model = self.model_type(train_inputs, train_targets, self.likelihood)
        # Extract dimensions for external use.
        self.input_dimension = train_inputs.shape[1]
        self.output_dimension = target_dimension
        self.n_training_samples = train_inputs.shape[0]

    def _compute_GP_covariances(self, train_x):
        """Compute K(X,X) + sigma*I and its inverse.

        """
        # Pre-compute inverse covariance plus noise to speed-up computation.
        K_lazy = self.model.covar_module(train_x.double())
        K_lazy_plus_noise = K_lazy.add_diag(self.model.likelihood.noise)
        n_samples = train_x.shape[0]
        self.model.K_plus_noise = K_lazy_plus_noise.matmul(torch.eye(n_samples).double())
        self.model.K_plus_noise_inv = K_lazy_plus_noise.inv_matmul(torch.eye(n_samples).double())  # self.model.K_plus_noise_inv_2 = torch.inverse(self.model.K_plus_noise) # Equivalent to above but slower.

    def init_with_hyperparam(self, train_inputs, train_targets, path_to_statedict):
        """Load hyperparameters from a state_dict.

        """
        if self.input_mask is not None:
            train_inputs = train_inputs[:, self.input_mask]
        if self.target_mask is not None:
            train_targets = train_targets[:, self.target_mask]
        device = torch.device('cpu')
        state_dict = torch.load(path_to_statedict, map_location=device)
        self._init_model(train_inputs, train_targets)
        if self.NORMALIZE:
            train_inputs = torch.from_numpy(self.scaler.transform(train_inputs.numpy()))
        self.model.load_state_dict(state_dict)
        self.model.double()  # needed otherwise loads state_dict as float32
        self._compute_GP_covariances(train_inputs)
        self.casadi_predict = self.make_casadi_prediction_func(train_inputs, train_targets)

    def train(self,
              train_input_data,
              train_target_data,
              test_input_data,
              test_target_data,
              n_train=500,
              learning_rate=0.01,
              gpu=False,
              fname='best_model.pth', ):
        """Train the GP using Train_x and Train_y.

        Args:
            train_x: Torch tensor (N samples [rows] by input dim [cols])
            train_y: Torch tensor (N samples [rows] by target dim [cols])

        """
        train_x_raw = train_input_data
        train_y_raw = train_target_data
        test_x_raw = test_input_data
        test_y_raw = test_target_data
        if self.input_mask is not None:
            train_x_raw = train_x_raw[:, self.input_mask]
            test_x_raw = test_x_raw[:, self.input_mask]
        if self.target_mask is not None:
            train_y_raw = train_y_raw[:, self.target_mask]
            test_y_raw = test_y_raw[:, self.target_mask]
        self._init_model(train_x_raw, train_y_raw)
        if self.NORMALIZE:
            train_x = torch.from_numpy(self.scaler.transform(train_x_raw))
            test_x = torch.from_numpy(self.scaler.transform(test_x_raw))
            train_y = train_y_raw
            test_y = test_y_raw
        else:
            train_x = train_x_raw
            train_y = train_y_raw
            test_x = test_x_raw
            test_y = test_y_raw
        if gpu:
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            test_x = test_x.cuda()
            test_y = test_y.cuda()
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
        self.model.double()
        self.likelihood.double()
        self.model.train()
        self.likelihood.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        last_loss = 99999999
        best_loss = 99999999
        loss = torch.tensor(0)
        i = 0
        while i < n_train and torch.abs(loss - last_loss) > 1e-2:
            with torch.no_grad():
                self.model.eval()
                self.likelihood.eval()
                test_output = self.model(test_x)
                test_loss = -mll(test_output, test_y)
            self.model.train()
            self.likelihood.train()
            self.optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            if i % 100 == 0:
                print('Iter %d/%d - MLL trian Loss: %.3f, Posterior Test Loss: %0.3f' % (
                    i + 1, n_train, loss.item(), test_loss.item()))

            self.optimizer.step()
            # if test_loss < best_loss:
            #    best_loss = test_loss
            #    state_dict = self.model.state_dict()
            #    torch.save(state_dict, fname)
            #    best_epoch = i
            if loss < best_loss:
                best_loss = loss
                state_dict = self.model.state_dict()
                torch.save(state_dict, fname)
                best_epoch = i

            i += 1
        print("Training Complete")
        print("Lowest epoch: %s" % best_epoch)
        print("Lowest Loss: %s" % best_loss)
        self.model = self.model.cpu()
        self.likelihood = self.likelihood.cpu()
        train_x = train_x.cpu()
        train_y = train_y.cpu()
        self.model.load_state_dict(torch.load(fname))
        self._compute_GP_covariances(train_x)
        self.casadi_predict = self.make_casadi_prediction_func(train_x, train_y)

    def predict(self, x, requires_grad=False, return_pred=True):
        """

        Args:
            x : torch.Tensor (N_samples x input DIM).

        Returns:
            Predictions
                mean : torch.tensor (nx X N_samples).
                lower : torch.tensor (nx X N_samples).
                upper : torch.tensor (nx X N_samples).

        """
        self.model.eval()
        self.likelihood.eval()
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).double()
        if self.input_mask is not None:
            x = x[:, self.input_mask]
        if self.NORMALIZE:
            x = torch.from_numpy(self.scaler.transform(x))
        if requires_grad:
            predictions = self.likelihood(self.model(x))
            mean = predictions.mean
            cov = predictions.covariance_matrix
        else:
            with torch.no_grad(), gpytorch.settings.fast_pred_var(state=True), gpytorch.settings.fast_pred_samples(state=True):
                predictions = self.likelihood(self.model(x))
                mean = predictions.mean
                cov = predictions.covariance_matrix
        if return_pred:
            return mean, cov, predictions
        else:
            return mean, cov

    def prediction_jacobian(self, query):
        mean_der, cov_der = torch.autograd.functional.jacobian(lambda x: self.predict(x,
                                                                                      requires_grad=True,
                                                                                      return_pred=False),
                                                               query.double())
        return mean_der.detach().squeeze()

    def make_casadi_prediction_func(self, train_inputs, train_targets):
        """
        Assumes train_inputs and train_targets are already masked.
        """
        train_inputs = train_inputs.numpy()
        train_targets = train_targets.numpy()
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach().numpy()
        output_scale = self.model.covar_module.outputscale.detach().numpy()
        Nx = len(self.input_mask)
        z = cs.SX.sym('z', Nx)
        K_z_ztrain = cs.Function('k_z_ztrain',
                                 [z],
                                 [covSEard(z, train_inputs.T, lengthscale.T, output_scale)],
                                 ['z'],
                                 ['K'])
        predict = cs.Function('pred',
                              [z],
                              [K_z_ztrain(z=z)['K'] @ self.model.K_plus_noise_inv.detach().numpy() @ train_targets],
                              ['z'],
                              ['mean'])
        return predict

    def plot_trained_gp(self, inputs, targets, output_label, fig_count=0):
        if self.target_mask is not None:
            targets = targets[:, self.target_mask]
        means, covs, preds = self.predict(inputs)
        t = np.arange(inputs.shape[0])
        lower, upper = preds.confidence_region()
        for i in range(self.output_dimension):
            fig_count += 1
            plt.figure(fig_count)
            if lower.ndim > 1:
                plt.fill_between(t, lower[:, i].detach().numpy(), upper[:, i].detach().numpy(), alpha=0.5, label='95%')
                plt.plot(t, means[:, i], 'r', label='GP Mean')
                plt.plot(t, targets[:, i], '*k', label='Data')
            else:
                plt.fill_between(t, lower.detach().numpy(), upper.detach().numpy(), alpha=0.5, label='95%')
                plt.plot(t, means, 'r', label='GP Mean')
                plt.plot(t, targets, '*k', label='Data')
            plt.legend()
            plt.title('Fitted GP x%s' % output_label)
            plt.xlabel('Time (s)')
            plt.ylabel('v')
            plt.show()
        return fig_count


def kmeans_centriods(n_cent, data, rand_state=0):
    """kmeans clustering. Useful for finding reasonable inducing points.

    Args:
        n_cent (int): Number of centriods.
        data (np.array): Data to find the centroids of n_samples X n_features.

    Return:
        centriods (np.array): Array of centriods (n_cent X n_features).

    """
    kmeans = KMeans(n_clusters=n_cent, random_state=rand_state).fit(data)
    return kmeans.cluster_centers_


class DataHandler:
    @classmethod
    def load(cls, load_dir):
        shelf_name = os.path.join(load_dir, 'data_handler.out')
        init_dict = {}
        with shelve.open(shelf_name) as myshelf:
            for key in myshelf:
                #setattr(cls, key, myshelf['key'])
                init_dict[key] = myshelf[key]
            myshelf.close()
        prior_model_name = os.path.join(load_dir, 'prior_model.casadi')
        init_dict['prior_model'] = cs.Function.load(prior_model_name)
        new_class = cls(**init_dict)
        return new_class

    def __init__(self,
                 x_data: np.array=None,
                 u_data: np.array=None,
                 prior_model: cs.Function=None,
                 save_dir: str=None,
                 train_test_ratio: float=0.8,
                 noise: dict=None,
                 normalize_inputs=False,
                 normalize_outputs=False,
                 num_samples=None,
                 seed=42):

        self.x_data = x_data
        self.u_data = u_data
        self.prior_model = prior_model
        self.train_test_ratio = train_test_ratio
        self.num_samples = num_samples
        self.seed = seed
        self.noise = noise
        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        self.save_dir = save_dir
        self.save_dict = deepcopy(self.__dict__)
        self.data = None
        self._initialize()

    def _initialize(self):
        rand_generator = np.random.default_rng(self.seed)

        if isinstance(self.x_data, list):
            inputs = []
            targets = []
            x_seq = []
            u_seq = []
            x_next_seq = []
            for i in range(len(self.x_data)):
                inputs_i, targets_i, x_seq_i, u_seq_i, x_next_seq_i = trajectory_to_training_data(self.x_data[i],
                                                                                                  self.u_data[i],
                                                                                                  self.prior_model)
                inputs.append(inputs_i)
                targets.append(targets_i)
                x_seq.append(x_seq_i)
                u_seq.append(u_seq_i)
                x_next_seq.append(x_next_seq_i)
            inputs = np.vstack(inputs)
            targets = np.vstack(targets)
        else:
            inputs, targets, x_seq, u_seq, x_next_seq = trajectory_to_training_data(self.x_data,
                                                                                    self.u_data,
                                                                                    self.prior_model)
            x_seq = [x_seq]
            u_seq = [u_seq]
            x_next_seq = [x_next_seq]

        if self.num_samples is not None and self.num_samples < inputs.shape[0]:
            interval = int(np.ceil(inputs.shape[0]/self.num_samples))
        else:
            interval = 1
        train_inputs, train_targets, test_inputs, test_targets = make_train_and_test_sets(inputs[::interval,:],
                                                                                          targets[::interval,:],
                                                                                          ratio=self.train_test_ratio,
                                                                                          rand_generator=rand_generator)
        if self.normalize_inputs or self.normalize_outputs:
            raise NotImplementedError("Normalization is still TBC.")

        if self.noise is not None:
            train_targets = add_noise(train_targets, self.noise, rand_generator)
            test_targets = add_noise(test_targets, self.noise, rand_generator)

        self.data = {'train_inputs': train_inputs,
                     'train_targets': train_targets,
                     'test_inputs': test_inputs,
                     'test_targets': test_targets,
                     'x_seq': x_seq,
                     'u_seq': u_seq,
                     'x_next_seq': x_next_seq}
        self.data = munchify(self.data)

    def save(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        full_dir = os.path.join(save_dir, 'data_handler')
        mkdirs(full_dir)
        fname = os.path.join(full_dir, 'data_handler.out')
        with shelve.open(fname) as myshelf:
            for kw, val in self.save_dict.items():
                if not('prior_model' == kw):
                    myshelf[kw] = val
            myshelf.close()
        cs_fname = os.path.join(full_dir, 'prior_model.casadi')
        self.prior_model.save(cs_fname)

    def select_subsamples_with_kmeans(self, n_sub, seed):
        centroids = kmeans_centriods(n_sub, self.data.train_inputs, rand_state=seed)

        contiguous_masked_inputs = np.ascontiguousarray(self.data.train_inputs) # required for version sklearn later than 1.0.2
        inds, dist_mat = pairwise_distances_argmin_min(centroids, contiguous_masked_inputs)
        input_data = self.data.train_inputs[inds]
        target_data = self.data.train_targets[inds]
        return input_data, target_data


def trajectory_to_training_data(x_data, u_data, prior_model):
        x_seq, u_seq, x_next_seq = gather_training_samples(x_data,
                                                           u_data)

        inputs, targets = make_inputs_and_targets(x_seq, u_seq, x_next_seq, prior_model)

        return inputs, targets, x_seq, u_seq, x_next_seq

def gather_training_samples(x_data: np.array,
                            u_data: np.array):
    """
    Preprocesses the data into states, next_states, and inputs, and processthem accordingly.

    Args:
        x_data: n_data+1 X dim(x) state data
        u_data: n_data X 1 input data (for single input only for now)
        num_samples: integer for the desired number of samples to take. If None, take them all
        rand_generator: random number generator for repeatability.

    Returns:
        x_seq: num_samples X dim(x) for the initial states
        u_seq: num_samples X dim(u) for the inputs at x_seq
        x_next_seq: num_samples X dim(x) for the next states.
    """
    n = u_data.shape[0]
    rand_inds_int = np.arange(0, n)
    next_inds_int = rand_inds_int + 1
    x_seq = x_data[rand_inds_int, :]
    u_seq = u_data[rand_inds_int, :]
    x_next_seq = x_data[next_inds_int, :]

    return x_seq, u_seq, x_next_seq

def make_inputs_and_targets(x_seq,
                            u_seq,
                            x_next_seq,
                            prior_model):
    """Converts trajectory data for GP trianing.
    Assumes equilibrium is all zeros.

    Args:
        x_seq (list): state sequence of np.array (nx,).
        u_seq (list): action sequence of np.array (nu,).
        x_next_seq (list): next state sequence of np.array (nx,).

    Returns:
        np.array: inputs for GP training, (N, nx+nu).
        np.array: targets for GP training, (N, nx).

    """
    # Get the predicted dynamics. This is a linear prior, thus we need to account for the fact that
    # it is linearized about an eq using self.X_GOAL and self.U_GOAL.
    x_pred_seq = prior_model(x0=x_seq.T , p=u_seq.T)['xf'].toarray()
    targets = (x_next_seq.T - x_pred_seq).transpose()  # (N, nx).
    inputs = np.hstack([x_seq, u_seq])  # (N, nx+nu).
    return inputs, targets

def make_train_and_test_sets(inputs, targets, ratio, rand_generator):
    train_idx, test_idx = train_test_split(
        list(range(inputs.shape[0])),
        train_size=ratio,
        random_state=np.random.RandomState(rand_generator.bit_generator)
    )
    train_inputs = inputs[train_idx, :]
    train_targets = targets[train_idx, :]
    test_inputs = inputs[test_idx, :]
    test_targets = targets[test_idx, :]
    return train_inputs, train_targets, test_inputs, test_targets

def add_noise(x_array, noise, rand_generator):
    """
    Add noise to x_array.
    Args:
        x_array: (np.array) N x nx array
        noise: (dict) Dictionary with mean and std for each dim
        rand_generator: numper random generator

    Returns:
        noisy_x_array
    """
    noisy_x_array = x_array + rand_generator.normal(loc=noise['mean'], scale=noise['std'], size=x_array.shape)
    return noisy_x_array

def combine_prior_and_gp(prior_model, gp_predict, input_mask, target_mask,):

    x = cs.SX.sym('x', 3)
    u = cs.SX.sym('u', 1)
    z = cs.vertcat(x, u)
    z = z[input_mask,:]
    Bd = make_bd(3, target_mask)

    x_next = prior_model(x0=x, p=u)['xf'] + Bd @ gp_predict(z=z)['mean']

    combined_dyn = cs.Function('dyn_func',
                               [x, u],
                               [x_next],
                               ['x0', 'p'],
                               ['xf'])
    return combined_dyn

def make_bd(n_sys, tar_mask):
    Bd = np.zeros((n_sys, len(tar_mask)))
    for input_ind, output_ind in enumerate(tar_mask):
        Bd[output_ind, input_ind] = 1

    return Bd

def get_LHS_samples(lower_bounds=None, upper_bounds=None, num_samples=None, seed=42):
    sampler = qmc.LatinHypercube(d=len(upper_bounds), seed=seed)
    samples = sampler.random(n=num_samples)
    scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)

    return scaled_samples

def get_MVN_samples(means, cov, num_samples, seed=42):
    sampler = qmc.MultivariateNormalQMC(mean=means, cov=cov)
    samples = sampler.random(num_samples)

    return samples

def get_next_real_states(x, u, quad):
    x_next = quad.cs_true_flat_dyn_from_x_and_u(x=x, u=u)['x_next'].toarray()
    return x_next

def generate_samples_into_sequences(sampler, sampler_args, quad):
    # LHS params
    xu_seqs = sampler(**sampler_args)
    xs = xu_seqs[:, :3].T
    us = xu_seqs[None, :,-1]
    x_nexts = get_next_real_states(xs, us, quad)
    x_data = []
    u_data = []
    for i in range(xs.shape[1]):
        if not(any(np.isnan(x_nexts[:,i]))):
            x_data.append(np.vstack((xs[:,i], x_nexts[:,i])))
            u_data.append(us[:,i,None])
    return x_data, u_data

def kmeans_centriods(n_cent, data, rand_state=0):
    """kmeans clustering. Useful for finding reasonable inducing points.

    Args:
        n_cent (int): Number of centriods.
        data (np.array): Data to find the centroids of n_samples X n_features.

    Return:
        centriods (np.array): Array of centriods (n_cent X n_features).

    """
    kmeans = KMeans(n_clusters=n_cent, random_state=rand_state).fit(data)
    return kmeans.cluster_centers_
