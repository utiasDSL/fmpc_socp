import GPy
import numpy as np
import gpytorch
import torch
import matplotlib.pyplot as plt
from gpytorch.constraints import Positive

def weighted_distance(x1 : torch.Tensor, x2 : torch.Tensor, L : torch.Tensor) -> torch.Tensor:
    """Computes (x1-x2)^T L (x1-x2)
    Args:
        x1 (torch.tensor) : N x n tensor (N is number of samples and n is dimension of vector)
        x2 (torch.tensor) : M x n tensor
        L (torch.tensor) : nxn wieght matrix

    Returns:
        weighted_distances (torch.tensor) : N x M tensor of all the prossible products

    """
    N = x1.shape[0]
    M = x2.shape[0]
    n = L.shape[0]
    # subtract all vectors in x2 from all vectors in x1 (results in NxMxn matrix)
    diff = x1.unsqueeze(1) - x2
    diff_T = diff.reshape(N,M,n,1)
    diff = diff.reshape(N,M,1,n)
    L = L.reshape(1,1,n,n)
    L = L.repeat(N,M,1,1)
    weighted_distances = torch.matmul(diff, torch.matmul(L,diff_T))
    return weighted_distances.squeeze()

def squared_exponential(x1,x2,L,var):
    return var * torch.exp(-0.5 * weighted_distance(x1, x2, L))

class AffineKernel(gpytorch.kernels.Kernel):
    is_stationary = False
    def __init__(self, input_dim,
                 length_prior=None,
                 length_constraint=None,
                 variance_prior=None,
                 variance_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.register_parameter(
            name='raw_length', parameter=torch.nn.Parameter(torch.zeros(2*(self.input_dim-1)))
        )
        self.register_parameter(
            name='raw_variance', parameter=torch.nn.Parameter(torch.zeros(2))
        )
        # set the parameter constraint to be positive, when nothing is specified
        if length_constraint is None:
            length_constraint = Positive()
        if variance_constraint is None:
            variance_constraint = Positive()
        # register the constraints
        self.register_constraint("raw_length", length_constraint)
        self.register_constraint("raw_variance", variance_constraint)
        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if length_prior is not None:
            self.register_prior(
                "length_prior",
                length_prior,
                lambda m: m.length,
                lambda m, v: m._set_length(v),
            )
        if variance_prior is not None:
            self.register_prior(
                "variance_prior",
                variance_prior,
                lambda m: m.variance,
                lambda m, v: m._set_variance(v),
            )

    # now set up the 'actual' paramter
    @property
    def length(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_length_constraint.transform(self.raw_length)

    @length.setter
    def length(self, value):
        return self._set_length(value)

    def _set_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_length=self.raw_length_constraint.inverse_transform(value))


    @property
    def variance(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value):
        return self._set_variance(value)

    def _set_variance(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_variance)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_variance=self.raw_variance_constraint.inverse_transform(value))

    def kappa_alpha(self,x1, x2):
        L_alpha = torch.diag(1 / (self.length[0:self.input_dim-1] ** 2))
        var_alpha = self.variance[0]
        return squared_exponential(x1, x2, L_alpha, var_alpha)

    def kappa_beta(self, x1, x2):
        L_beta = torch.diag(1 / (self.length[self.input_dim-1:] ** 2))
        var_beta = self.variance[1]
        return squared_exponential(x1, x2, L_beta, var_beta)

    def kappa(self,x1, x2):
        z1 = x1[:, 0:self.input_dim - 1]
        u1 = x1[:, -1, None]
        z2 = x2[:, 0:self.input_dim - 1]
        u2 = x2[:, -1, None]
        u_mat = u1.unsqueeze(1) * u2
        kappa = self.kappa_alpha(z1, z2) + self.kappa_beta(z1, z2) * u_mat.squeeze()
        if kappa.dim() < 2:
            return kappa.unsqueeze(0)
        else:
            return kappa
    # this is the kernel function
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
        kern = self.kappa(x1, x2)
        if diag:
            return kern.diag()
        else:
            return kern

class AffineGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        """Zero mean with Affine Kernel GP model for SISO systems

        Args:
            train_x (torch.Tensor): input training data (N_samples x input_dim)
            train_y (torch.Tensor): output training data (N_samples x 1)
            likelihood (gpytorch.likelihood): Likelihood function
                (gpytorch.likelihoods.MultitaskGaussianLikelihood)
        """
        super().__init__(train_x, train_y, likelihood)
        self.input_dim = train_x.shape[1]
        #self.output_dim = train_y.shape[1]
        self.output_dim = 1
        self.n = 1     #output dimension
        #self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = None
        self.covar_module = AffineKernel(self.input_dim)
        self.K_plus_noise_inv = None

    def forward(self, x):
        mean_x = self.mean_module(x) # is this needed for ZeroMean?
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def compute_gammas(self, query):
        return NotImplementedError

    def mean_and_cov_from_gammas(self,query):
        gamma_1, gamma_2, gamma_3, gamma_4, gamma_5 = self.compute_gammas(query)
        u = query[:, None, 1]
        means_from_gamma = gamma_1 + gamma_2.mul(u)
        covs_from_gamma = gamma_3 + gamma_4.mul(u) + gamma_5.mul(u ** 2) + self.likelihood.noise.detach()
        upper_from_gamma = means_from_gamma + covs_from_gamma.sqrt() * 2
        lower_from_gamma = means_from_gamma - covs_from_gamma.sqrt() * 2
        return means_from_gamma, covs_from_gamma, upper_from_gamma, lower_from_gamma

class ZeroMeanAffineGP(AffineGP):
    def __init__(self, train_x, train_y, likelihood):
        """Zero mean with Affine Kernel GP model for SISO systems

        Args:
            train_x (torch.Tensor): input training data (N_samples x input_dim)
            train_y (torch.Tensor): output training data (N_samples x 1)
            likelihood (gpytorch.likelihood): Likelihood function
                (gpytorch.likelihoods.MultitaskGaussianLikelihood)
        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

    def compute_gammas(self, query):
        # Parse inputs
        with torch.no_grad():
            n_train_samples = self.train_targets.shape[0]
            n_query_samples = query.shape[0]
            zq = query[:,0:self.input_dim-1]
            uq = query[:,-1,None]
            z_train = self.train_inputs[0][:,0:self.input_dim-1]
            u_train = self.train_inputs[0][:,-1,None].tile(n_query_samples)
            # Precompute useful matrics
            k_a = self.covar_module.kappa_alpha(zq, z_train)
            if k_a.dim() == 1:
                k_a = k_a.unsqueeze(0)
            k_b = self.covar_module.kappa_beta(zq, z_train).mul(u_train.T)
            if k_b.dim() == 1:
                k_b = k_b.unsqueeze(0)
            Psi = self.train_targets.reshape((n_train_samples,1))
            # compute gammas (Note: inv_matmul(R, L) = L * inv(K) * R
            gamma_1 = k_a @ self.K_plus_noise_inv @ Psi
            gamma_2 = k_b @ self.K_plus_noise_inv @ Psi
            gamma_3 = torch.diag(self.covar_module.kappa_alpha(zq,zq) - k_a @ self.K_plus_noise_inv @ k_a.T)
            gamma_4 = torch.diag(-( k_b @ self.K_plus_noise_inv @ k_a.T + k_a @ self.K_plus_noise_inv @ k_b.T))
            gamma_5 = torch.diag(self.covar_module.kappa_beta(zq,zq) - k_b @ self.K_plus_noise_inv @ k_b.T)
        return gamma_1, gamma_2, gamma_3.unsqueeze(1), gamma_4.unsqueeze(1), gamma_5.unsqueeze(1)

class ConstantMeanAffineGP(AffineGP):
    def __init__(self, train_x, train_y, likelihood, mean_prior=None):
        """Zero mean with Affine Kernel GP model for SISO systems

        Args:
            train_x (torch.Tensor): input training data (N_samples x input_dim)
            train_y (torch.Tensor): output training data (N_samples x 1)
            likelihood (gpytorch.likelihood): Likelihood function
                (gpytorch.likelihoods.MultitaskGaussianLikelihood)
            mean_prior (NOT SURE) : prior on the constant mean
        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

    def compute_gammas(self, query):
        # Parse inputs
        with torch.no_grad():
            n_train_samples = self.train_targets.shape[0]
            n_query_samples = query.shape[0]
            zq = query[:,0:self.input_dim-1]
            uq = query[:,-1,None]
            z_train = self.train_inputs[0][:,0:self.input_dim-1]
            u_train = self.train_inputs[0][:,-1,None].tile(n_query_samples)
            # Precompute useful matrics
            k_a = self.covar_module.kappa_alpha(zq, z_train)
            if k_a.dim() == 1:
                k_a = k_a.unsqueeze(0)
            k_b = self.covar_module.kappa_beta(zq, z_train).mul(u_train.T)
            if k_b.dim() == 1:
                k_b = k_b.unsqueeze(0)
            Psi = self.train_targets.reshape((n_train_samples,1))
            # compute gammas (Note: inv_matmul(R, L) = L * inv(K) * R
            gamma_1 = k_a @ self.K_plus_noise_inv @ (Psi - u_train*self.mean_module.constant)
            gamma_2 = self.mean_module.constant + k_b @ self.K_plus_noise_inv @ (Psi - u_train*self.mean_module.constant)
            gamma_3 = torch.diag(self.covar_module.kappa_alpha(zq,zq) - k_a @ self.K_plus_noise_inv @ k_a.T)
            gamma_4 = torch.diag(-( k_b @ self.K_plus_noise_inv @ k_a.T + k_a @ self.K_plus_noise_inv @ k_b.T))
            gamma_5 = torch.diag(self.covar_module.kappa_beta(zq,zq) - k_b @ self.K_plus_noise_inv @ k_b.T)
        return gamma_1, gamma_2, gamma_3.unsqueeze(1), gamma_4.unsqueeze(1), gamma_5.unsqueeze(1)

class GaussianProcess():
    def __init__(self, model_type, likelihood, n):
        """
        Gaussian Process decorator for gpytorch
        Args:
            model_type (gpytorch model class): Model class for the GP (ZeroMeanIndependentMultitaskGPModel)
            likelihood (gpytorch.likelihood): likelihood function
            n (int): Dimension of input state space

        """
        self.model_type = model_type
        self.likelihood = likelihood
        self.m = n
        self.optimizer = None
        self.model = None

    def train(self, train_x, train_y, n_train=150):
        """
        Train the GP using Train_x and Train_y
        train_x: Torch tensor (dim input x N samples)
        train_y: Torch tensor (nx x N samples)

        """
        self.n = train_x.shape[1]
        self.m = 1
        self.output_dim = 1
        self.input_dim = train_x.shape[1]
        if self.model is None:
            self.model = self.model_type(train_x, train_y, self.likelihood)
        else:
            train_x = torch.reshape(train_x, self.model.train_inputs[0].shape)
            train_x = torch.cat([train_x, self.model.train_inputs[0]])
            train_y = torch.cat([train_y, self.model.train_targets])
            self.model.set_train_data(train_x, train_y, False)
        self.model.double()
        self.likelihood.double()
        self.model.train()
        self.likelihood.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.5)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(n_train):
            self.optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, n_train, loss.item()))
            self.optimizer.step()
        # compute inverse covariance plus noise for faster computation later
        K_lazy = self.model.covar_module(train_x.double())
        K_lazy_plus_noise = K_lazy.add_diag(self.model.likelihood.noise)
        n_samples = train_x.shape[0]
        self.model.K_plus_noise_inv = K_lazy_plus_noise.inv_matmul(torch.eye(n_samples).double())

    def predict(self, x, requires_grad=False, return_pred=True):
        """
        x : torch.Tensor (input dim X N_samples)
        Return
            Predicitons
            mean : torch.tensor (nx X N_samples)
            lower : torch.tensor (nx X N_samples)
            upper : torch.tensor (nx X N_samples)
        """
        #x = torch.from_numpy(x).double()
        self.model.eval()
        self.likelihood.eval()
        #with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).double()

        if requires_grad:
            predictions = self.likelihood(self.model(x))
            mean = predictions.mean
            cov = predictions.covariance_matrix
        else:
            with torch.no_grad():
                predictions = self.likelihood(self.model(x))
                mean = predictions.mean
                cov = predictions.covariance_matrix
        if return_pred:
            return mean, cov, predictions
        else:
            return mean, cov

    def prediction_jacobian(self, query):
        gammas = self.model.compute_gammas(query)
        mean_der = gammas[1]
        cov_der = gammas[4]
        #mean_der, _ = torch.autograd.functional.jacobian(
        #                        lambda x: self.predict(x, requires_grad=True, return_pred=False),
        #                        query.double())
        #k_query_query = torch.autograd.functional.hessian(
        #                               lambda x: self.model.covar_module.kappa(x,x), query.double()
        #)
        #k_v_v = k_query_query.squeeze()[-1,-1]
        #k_a_prime = torch.autograd.functional.jacobian(
        #        lambda x: self.model.covar_module.kappa(x, self.model.train_inputs[0]), query.double()
        #)
        #k_a = k_a_prime.squeeze()[:,-1,None]
        #cov_der = k_v_v - k_a.T @ self.model.K_plus_noise_inv @ k_a #+ self.model.likelihood.noise

        #k_v_v = self.model.covar_module.kappa_beta(query[:,None,0:3], query[:,None,0:3])
        #u_train = self.model.train_inputs[0][:, -1, None]
        #k_b = self.model.covar_module.kappa_beta(query[:,None,0:3], self.model.train_inputs[0][:,0:3]).mul(u_train.T)
        #if k_b.dim() == 1:
        #    k_b = k_b.unsqueeze(0)
        ##k_b = self.model.covar_module.kappa_beta(query[:,None,0:3],self.model.train_inputs[0][:,0:3]).unsqueeze(0)
        #cov_der = k_v_v - k_b @ self.model.K_plus_noise_inv @ k_b.T  #+ self.model.likelihood.noise
        #cov_der = k_v_v

        return mean_der.detach(), cov_der.detach()

    def plot_trained_gp(self, t, fig_count=0):
        means, covs, preds = self.predict(self.model.train_inputs[0])
        lower, upper = preds.confidence_region()
        fig_count += 1
        plt.figure(fig_count)
        plt.fill_between(t, lower.detach().numpy(), upper.detach().numpy(), alpha=0.5, label='95%')
        plt.plot(t, means, 'r', label='GP Mean')
        plt.plot(t, self.model.train_targets, '*k', label='Data')
        plt.legend()
        plt.title('Fitted GP')
        plt.xlabel('Time (s)')
        plt.ylabel('v')
        plt.show()

        return fig_count


def train_gp(noise, variance_input, n, gp_input, gp_output):
    gp_kernel = GPy.kern.RBF(input_dim=n, ARD=True, variance=variance_input, lengthscale=1.0)

    gp_model = GPy.models.GPRegression(gp_input, gp_output, gp_kernel, normalizer=None, noise_var=noise)
    gp_model.rbf.variance.constrain_fixed()
    gp_model.Gaussian_noise.variance.constrain_fixed(noise)
    gp_model.optimize(max_iters=1000, optimizer='bfgs', messages=True)

    return gp_model

class SISOAffineGP:
    def __init__(self, noise, variance_input, n, gp_input, gp_output):
        self.model = self.train_gp(noise, variance_input, n, gp_input, gp_output)
        self.output_dim = self.model.output_dim
        self.input_dim = self.model.input_dim
        self.n_samples = gp_input.shape[0]

        self.create_affine_kernels()

        self.X = gp_input[:,0:3]
        self.U = gp_input[:,-1,np.newaxis]
        self.UU = self.U @ self.U.T

        self.input = gp_input
        self.output = gp_output



        self.K = self.model.kern.K(gp_input) + np.eye(self.n_samples)*noise**2
        self.K_compute = self.kappa_alpha.K(self.X) + np.multiply(self.UU, self.kappa_beta.K(self.X)) + np.eye(self.n_samples)*noise**2
        self.K_inv = np.linalg.inv(self.K)
        self.Psi = gp_output

        self.noise = noise
        self.variance = variance_input



    def train_gp(self, noise, variance_input, n, gp_input, gp_output):
        k_alpha = GPy.kern.RBF(input_dim=n-1, active_dims=[0,1,2], ARD=True, variance=variance_input, lengthscale=1.0,name='rbf_alpha')
        k_beta = GPy.kern.RBF(input_dim=n-1, active_dims=[0,1,2], ARD=True, variance=variance_input, lengthscale=1.0, name='rbf_beta')
        k_u = GPy.kern.Linear(input_dim=1, active_dims=[3], ARD=True, name='linear')


        k_prod = GPy.kern.Prod([k_u, k_beta])
        affine_kernel = GPy.kern.Add([k_alpha, k_prod])

        gp_model = GPy.models.GPRegression(gp_input, gp_output, affine_kernel, normalizer=False, noise_var=noise)
        gp_model.kern.parameters[1].parameters[0].variances.fix(1) # force linear kernel fix variance parameter
        gp_model.kern.parameters[0].variance.constrain_fixed()
        gp_model.kern.parameters[1].parameters[1].variance.constrain_fixed()
        gp_model.optimize(max_iters=10000, optimizer='bfgs', messages=True)

        return gp_model

    def create_affine_kernels(self):
        params = self.model.param_array
        var_alpha = params[0]
        l_alpha = params[1:4]
        var_beta = params[5]
        l_beta = params[6:-1]
        self.kappa_alpha = GPy.kern.RBF(input_dim=self.input_dim - 1,
                                        ARD=True,
                                        variance=var_alpha,
                                        lengthscale=l_alpha,
                                        name='kappa_alpha')
        self.kappa_beta = GPy.kern.RBF(input_dim=self.input_dim - 1,
                                        ARD=True,
                                        variance=var_beta,
                                        lengthscale=l_beta,
                                        name='kappa_beta')


    def predict(self,query):
        x = query[0,np.newaxis,0:self.input_dim-1]
        u = query[0,np.newaxis,-1,np.newaxis]
        k_alpha = self.kappa_alpha.K(x,self.X)
        k_beta = np.multiply(self.kappa_beta.K(x,self.X),self.U.T)

        gamma_1 = k_alpha @ self.K_inv @ self.Psi
        gamma_2 = k_beta @ self.K_inv @ self.Psi
        gamma_3 = self.kappa_alpha.K(x,x) - k_alpha @ self.K_inv @ k_alpha.T
        gamma_4 = -(k_beta @ self.K_inv @ k_alpha.T + k_alpha @ self.K_inv @ k_beta.T)
        gamma_5 = self.kappa_beta.K(x,x) - k_beta @ self.K_inv @ k_beta.T

        mean = gamma_1 + gamma_2*u
        cov = gamma_3 + gamma_4*u + gamma_5*u**2

        #cov_test = self.model.kern.K(query,query) - self.model.kern.K(query,self.input) @ np.linalg.inv(self.model.kern.K(self.input,self.input)) @ self.model.kern.K(self.input,query)
        #param_a = self.kappa_alpha.param_array
        #param_b = self.kappa_beta.param_array
        #test = affine_kernel(query,query,param_a, param_b) - \
        #        affine_kernel(query,self.input,param_a, param_b) @ np.linalg.inv(affine_kernel(self.input,self.input,param_a, param_b) + np.eye(500)*self.noise**2) @ affine_kernel(self.input, query,param_a, param_b)
        #mean, cov = self.model.predict(query)
        return mean, cov

def affine_kernel(z1, z2, params_a, params_b):
    variance_a = params_a[0]
    length_scales_a = params_a[1:]
    variance_b = params_b[1]
    length_scales_b = params_b[1:]

    x1 = z1[:,0:-1]
    x2 = z2[:,0:-1]
    k_a = se_kernel(x1, x2, variance_a, length_scales_a)
    k_b = se_kernel_u(z1, z2, variance_b, length_scales_b)


    k = k_a + k_b
    return k

def se_kernel_u(z1, z2, variance, length_scales):
    """
    x1 = Nsamples x input
    x2 = Nsamples x inputs
    length_scales : size of input
    """
    N1, n = z1.shape
    N2, n = z2.shape
    x1 = z1[:,0:-1]
    x2 = z2[:,0:-1]
    u1 = z1[:,-1]
    u2 = z2[:,-1]
    L_inv = np.diag(1/length_scales**2)
    val = np.zeros((N1,N2))
    for i in range(N1):
        for j in range(N2):
            val[i,j] = u1[i]*u2[j]*variance*np.exp(-0.5*(x1[np.newaxis,i,:].T-x2[np.newaxis,j,:].T).T @ L_inv @ (x1[np.newaxis,i,:].T - x2[np.newaxis,j,:].T))

    val = val

    return val

def se_kernel(x1, x2, variance, length_scales):
    """
    x1 = Nsamples x input
    x2 = Nsamples x inputs
    length_scales : size of input
    """
    N1, n = x1.shape
    N2, n = x2.shape
    L_inv = np.diag(1/length_scales**2)/2.0
    val = np.zeros((N1,N2))
    for i in range(N1):
        for j in range(N2):
            val[i,j] = variance*np.exp(-0.5*(x1[np.newaxis,i,:].T-x2[np.newaxis,j,:].T).T @ L_inv @ (x1[np.newaxis,i,:].T - x2[np.newaxis,j,:].T))
    val = val

    return val


