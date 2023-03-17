import numpy as np
from numpy.linalg import norm
import cvxpy as cp
from scipy.optimize import minimize
from robust_lqr import bound_computation
import torch
import GPy

class RobustLQR():
    def __init__(self, name, quad, Q, R, delta, bound=None, eps=1e-3, c0=100.0):
        self.name = name
        self.quad = quad
        self.quad.lqr_gain_and_ARE_soln(Q, R)
        self.K_gain = self.quad.K_gain
        self.P = self.quad.P
        self.S = self.quad.S
        self.B = self.quad.B
        self.prob_threshold = np.sqrt(1.0-delta)
        self.eps = eps
        self.c0 = c0
        self.bound = bound

    def compute_feedback_input(self, gp, z, z_ref, v_ref, x_init=None, **kwargs):
        v_des = -self.K_gain @ (z - z_ref) + v_ref
        query = np.hstack((z.T, v_des.T))
        if type(gp) == GPy.models.gp_regression.GPRegression:
            u_query = gp.predict(query)
            u_query_der = gp.predict_jacobian(query)  # this doesn't work for non-stationary kernels!
            mean = u_query[0][0]
            var = u_query[1][0]
            mean_der = u_query_der[0][0][-1]
            var_der = u_query_der[1][0][-1, -1]
        else:
            query = torch.from_numpy(query).double()
            # compute mean, variance and their derivatives wrt v
            mean, var, _ = gp.predict(query)
            mean = mean.numpy().squeeze()
            var = var.numpy().squeeze()
            mean_der, var_der = gp.prediction_jacobian(query)
            mean_der = mean_der.numpy().squeeze()
            var_der = var_der.numpy().squeeze()
        #mean_der = mean_der[-1]
        # compute bound inputs
        e = z-z_ref
        w = e.T@self.P@self.B
        w = w.squeeze()
        # compute bound
        c, success = bound_computation(0.0, 1.0+mean_der, var, var_der, 0.0, self.prob_threshold, self.c0)
        #if abs(w) > self.eps:
        #    v_rob = -c * np.sign(w)
        #else:
        #    v_rob = -c * w / self.eps
        v_rob = -c * np.sign(w)
        v = mean + v_rob + v_des
        # v = mean - c * np.sign(w) + v_des
        u = self.quad.u_from_v(v, z)
        if self.bound is not None:
            u = np.clip(u, -self.bound, self.bound)
        # saturate u to limits
        return u, v_des, success, 0


class LQR():
    def __init__(self, name, quad, Q, R, bound=None):
        self.name = name
        quad.lqr_gain_and_ARE_soln(Q, R)
        self.quad = quad
        self.K_gain = quad.K_gain
        self.P = quad.P
        self.S = quad.S
        self.bound = bound

    def compute_feedback_input(self, gp, z, z_ref, v_ref, x_init=None, **kwargs):
        v_des = - self.quad.K_gain.dot((z - z_ref)) + v_ref
        u = self.quad.u_from_v(v_des, z)
        if self.bound is not None:
            u = np.clip(u, -self.bound, self.bound)
        return u, v_des, True, 0


class ConstrainedQPProblem():
    def __init__(self, name, upper_bound, lower_bound, quad):
        self.name = name
        self.quad = quad
        self.p = cp.Parameter(shape=(1,1),pos=True)
        self.q = cp.Parameter(shape=(1,1))
        self.v_des = cp.Parameter(shape=(1,1))
        self.u = cp.Variable(shape=(1,1))
        self.lb = None
        self.ub = None

        # cost = (self.gamma2**2 + self.gamma5)*self.u**2 + \
        #(2 * self.gamma1 * self.gamma2 - 2 * self.gamma2 * self.v_des + self.gamma4) * self.u
        #cost = self.p*self.u**2  + self.q * self.u
        cost = self.p@self.u**2 + self.q @ self.u
        #cost = cp.QuadForm(self.u, self.p) + self.q @ self.u
        if upper_bound is not None and lower_bound is not None:
            self.lb = lower_bound <= self.u
            self.ub = self.u <= upper_bound
            prob = cp.Problem(cp.Minimize(cost),  # cost
                              [self.lb, self.ub])  # constraints
        else:
            prob = cp.Problem(cp.Minimize(cost))  # Only cost function
        self.prob = prob

    def solve(self, gp_model, z, v_des, x_init=None):
        query = np.hstack((z.T, np.zeros((1, 1))))
        query = torch.from_numpy(query).double()
        # compute gammas
        gamma1, gamma2, gamma3, gamma4, gamma5 = gp_model.model.compute_gammas(query)
        gamma1 = gamma1.numpy().squeeze()
        gamma2 = gamma2.numpy().squeeze()
        gamma3 = gamma3.numpy().squeeze()
        gamma4 = gamma4.numpy().squeeze()
        gamma5 = gamma5.numpy().squeeze()
        alpha_quad = self.quad.alpha(z)
        beta_quad = self.quad.beta(z)
        gamma1 = gamma1 + beta_quad.squeeze()
        gamma2 = gamma2 + alpha_quad.squeeze()
        #self.u = cp.Variable()
        #cost = (gamma2**2 + gamma5)*self.u**2 + (2*gamma1*gamma2 - 2*gamma2*v_des.squeeze() + gamma4)*self.u
        #self.prob = cp.Problem(cp.Minimize(self.cost) , [self.lb, self.ub])
        #print(gamma5)
        # assign parameters
        self.p.value = np.array([[gamma2**2 + gamma5]])
        self.q.value = np.array([[2 * gamma1 * gamma2 - 2 * gamma2 * v_des.squeeze() + gamma4]])
        #self.gamma1.value = gamma1
        #self.gamma2.value = gamma2
        #self.gamma3.value = gamma3
        #self.gamma4.value = gamma4
        #self.gamma5.value = gamma5
        #self.v_des.value = v_des.squeeze()

        #self.u.value = x_init[0]
        self.prob.solve(solver='MOSEK', warm_start=True, verbose=False)
        return self.u.value.squeeze()

    def compute_feedback_input(self, gp, z, z_ref, v_ref, x_init=None, **kwargs):
        """ Compute u so it can be used in feedback function"""
        v_des = - self.quad.K_gain.dot((z - z_ref)) + v_ref
        if self.lb is None and self.ub is None:
            u = unconstrained_closed_form_soln_for_difference_gp(gp, z, v_des, self.quad)
            success = True
        else:
            u = self.solve(gp, z, v_des, x_init=x_init)
            if 'infeasible' in self.prob.status:
                success = False
            else:
                success = True
        return u, v_des, success, 0


class SOCPProblem:
    def __init__(self, name, quad, beta, input_bound=None, state_bound=None, ctrl=None):
        self.name = name
        self.quad = quad
        self.beta = beta
        if quad.P is not None:
            self.P = quad.P
            self.S = quad.S
            self.Q = quad.Q
        else:
            Q = ctrl.Q
            R = ctrl.R
            self.quad.lqr_gain_and_ARE_soln(Q, R)
            self.P = quad.P
            self.S = quad.S
            self.Q = quad.Q
        self.input_bound = input_bound
        self.state_bound = state_bound
        self.s_min = np.min(np.linalg.eig(self.S)[0])
        if ctrl is not None:
            if not(ctrl.name == 'FMPC'):
                raise ValueError('Only FMPC can be used here for now.')
            self.ctrl = ctrl
        else:
            self.ctrl = None

        # Opt variables and parameters
        self.X = cp.Variable(shape=(3,))
        self.A1 = cp.Parameter(shape=(3, 3))
        self.A2 = cp.Parameter(shape=(3, 3))
        self.b1 = cp.Parameter(shape=(3,))
        self.b2 = cp.Parameter(shape=(3,))
        self.c1 = cp.Parameter(shape=(1, 3))
        self.c2 = cp.Parameter(shape=(1, 3))
        self.d1 = cp.Parameter()
        self.d2 = cp.Parameter()
        # put into lists
        As = [self.A1, self.A2]
        bs = [self.b1, self.b2]
        cs = [self.c1, self.c2]
        ds = [self.d1, self.d2]
        # Add input constraints if supplied
        if input_bound is not None:
            A3 = np.zeros((3, 3))
            A3[0, 0] = 1.0
            b3 = np.zeros((3, 1))
            c3 = np.zeros((1, 3))
            d3 = input_bound
            As.append(A3)
            bs.append(b3)
            cs.append(c3)
            ds.append(d3)
        if state_bound is not None:
            self.Astate = cp.Parameter(shape=(3, 3))
            self.bstate = cp.Parameter(shape=(3,))
            self.cstate = cp.Parameter(shape=(1, 3))
            self.dstate = cp.Parameter()
            #self.Astate = np.zeros((3,3))
            #self.Astate[0, 0] = 1.0
            #self.bstate = np.zeros((3, 1))
            #self.cstate = np.zeros((1, 3))
            #self.dstate = 0.0
            As.append(self.Astate)
            bs.append(self.bstate)
            cs.append(self.cstate)
            ds.append(self.dstate)
        else:
            self.Astate = None
            self.bstate = None
            self.cstate = None
            self.dstate = None
        # define cost function
        self.cost = cp.Parameter(shape=(1, 3))
        m = len(As)
        soc_constraints = [
            cp.SOC(cs[i] @ self.X + ds[i], As[i] @ self.X + bs[i]) for i in range(m)
        ]
        self.prob = cp.Problem(cp.Minimize(self.cost @ self.X), soc_constraints)

    def solve(self, gp_model, z, z_ref, v_des, x_init=np.zeros((3,))):
        e = z - z_ref
        cost, As, bs, cs, ds = socp_constraints_and_cost(gp_model,
                                                         z,
                                                         v_des,
                                                         self.beta,
                                                         e,
                                                         self.quad,
                                                         input_bound=self.input_bound,
                                                         state_bound=self.state_bound)
        self.cost.value = cost
        self.A1.value = As['A1']
        self.A2.value = As['A2']
        self.b1.value = bs['b1'].squeeze()
        self.b2.value = bs['b2'].squeeze()
        self.c1.value = cs['c1']
        self.c2.value = cs['c2']
        self.d1.value = ds['d1']
        self.d2.value = ds['d2']
        if self.state_bound is not None:
            self.Astate.value = As['Astate']
            self.bstate.value = bs['bstate'].squeeze()
            self.cstate.value = cs['cstate']
            self.dstate.value = ds['dstate']

        self.X.value = x_init
        self.prob.solve(solver='MOSEK', warm_start=True, verbose=True) # SCS was used in paper
        if 'optimal' in self.prob.status:
            return self.X.value[0], self.X.value[1]
        else:
            return 0, 0

    def compute_feedback_input(self, gp, z, z_ref, v_ref, x_init=None, t=None, params=None, **kwargs):
        """ Compute u so it can be used in feedback function"""
        if self.ctrl is None:
            v_des = - self.quad.K_gain.dot((z - z_ref)) + v_ref
            u, d_sf = self.solve(gp, z, z_ref, v_des, x_init=x_init)
        else:
            zd, v_des, return_status = self.ctrl.select_flat_input(z, t, params)
            zd = np.atleast_2d(zd).T
            u, d_sf = self.solve(gp, zd, z_ref, v_des, x_init=x_init)
        if 'optimal' in self.prob.status:
            success = True
        else:
            success = False
        return u, v_des, success, d_sf

def socp_constraints_and_cost(gp_model,
                              z,
                              v_des,
                              beta,
                              e,
                              quad,
                              input_bound=None,
                              state_bound=None,
                              d_weight=25,
                              eps=0.1):
    """ Setup the SOCP Opt constraint matrices for CVX.

    See following link for more information
    https://www.cvxpy.org/examples/basic/socp.html

    Args:
        gp_model (GaussianProcess) : GP model for nonlinear term (z,u) -> v
        z (np.array): current flat state
        v_des (np.array): 1x1 desired flat input
        beta (float): safety factor (number of stds)
        A (np.array): System state matrix A
        B (np.array): System input matrix B
        e (np.array): flat error state
        P (np.array): Soln to ARE (or DARE)
        z_ref_dot (np.array): time derivative of input reference
        s_min (float): smallest eigenvalue of S

    Returns:
        As (list): list of A socp constraint matrices
        bs (list): list of b socp constraint matrices
        cs (list): list of c socp constraint matrices
        ds (list): list of d socp constraint matrices
        cost (np.array): array of the cost matrix to be minimized

    """
    query_np = np.hstack((z.T, np.zeros((1, 1))))
    query = torch.from_numpy(query_np).double()
    gamma1, gamma2, gamma3, gamma4, gamma5 = gp_model.model.compute_gammas(query)
    gamma1 = gamma1.numpy().squeeze()
    gamma2 = gamma2.numpy().squeeze()
    gamma3 = gamma3.numpy().squeeze()
    gamma4 = gamma4.numpy().squeeze()
    gamma5 = gamma5.numpy().squeeze()
    B = quad.B
    w = e.T @ quad.P @ B
    w = w[0, 0]
    alpha_quad = quad.alpha(z)
    beta_quad = quad.beta(z)
    gamma1 = gamma1 + beta_quad.squeeze()
    gamma2 = gamma2 + alpha_quad.squeeze()
    gamma1 = gamma1.squeeze()
    gamma2 = gamma2.squeeze()
    norm_w = np.absolute(w)
    if state_bound is not None:
        h = state_bound['h']
        bcon = state_bound['b']
        phi_p = state_bound['phi_p']
        del_sig = phi_p * np.sqrt(h.T @ quad.Bd @ quad.Bd.T @ h)

    #if norm_w == 0:
    #    norm_w = 1e-6
        #w = 1e-6
    # create constraint matrics
    #A1 = np.array([[np.sqrt(gamma5), 0, 0],
    #               [0, 0, 0],
    #               [0, 0, 0]])
    #b1 = np.array([[gamma4 / (2 * np.sqrt(gamma5))],
    #               [np.sqrt(gamma3 - 0.25 * gamma4 ** 2 / gamma5)],
    #               [0]])
    A1 = np.array([[norm_w*np.sqrt(gamma5), 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    b1 = np.array([[norm_w*gamma4 / (2 * np.sqrt(gamma5))],
                   [norm_w*np.sqrt(gamma3 - 0.25 * gamma4 ** 2 / gamma5)],
                   [0]])
    #c1 = np.array([[-np.sign(w) * gamma2 / beta, 1 / 2*(norm_w * beta), 0]])
    #c1 = np.array([[-np.sign(w) * gamma2 / beta, 1 / 2*(norm_w * beta), 0]])
    #d1 = np.sign(w) * (v_des - gamma1)/beta + e.T @ (quad.S - quad.c3*quad.P) @ e / (2*norm_w*beta)
    c1 = np.array([[-w * gamma2 / beta, 1 / (2* beta), 0]])
    d1 = w * (v_des - gamma1)/beta + e.T @ (quad.S - quad.c3*quad.P) @ e / (2*beta)


    A2 = np.array([[2 * np.sqrt(gamma2 ** 2 + gamma5), 0, 0],
                   [0, 2 * d_weight, 0],
                   [0, 0, -1]])
    b2 = np.array([[0], [0], [1]])
    c2 = np.array([[0, 0, 1]])
    d2 = 1
    # put into lists
    As = {'A1': A1, 'A2': A2}
    bs = {'b1': b1, 'b2': b2}
    cs = {'c1': c1, 'c2': c2}
    ds = {'d1': d1.squeeze(), 'd2': d2}
    # Add input constraints if needed
    if input_bound is not None:
        A3 = np.zeros((3, 3))
        A3[0, 0] = 1.0
        b3 = np.zeros((3, 1))
        c3 = np.zeros((1, 3))
        d3 = input_bound
        As['A3'] = A3
        bs['b3'] = b3
        cs['c3'] = c3
        ds['d3'] = d3
    if state_bound is not None:
        Astate = np.array([[float(del_sig*np.sqrt(gamma5)), 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])
        bstate = np.array([[float(del_sig*gamma4 / (2 * np.sqrt(gamma5)))],
                           [float(del_sig*np.sqrt(gamma3 - 0.25 * gamma4 ** 2 / gamma5))],
                           [0]])
        cstate = np.array([[float(-h.T @ quad.Bd * gamma2), 0.0, 0.0]])
        dstate = -h.T @ quad.Ad @ z - h.T @ quad.Bd * gamma1 + bcon
        As['Astate'] = Astate
        bs['bstate'] = bstate
        cs['cstate'] = cstate
        ds['dstate'] = float(dstate)

    # define cost matrix
    cost = np.array([[2 * gamma1 * gamma2 - 2 * gamma2 * v_des.squeeze() + gamma4, 0, 1]])
    return cost, As, bs, cs, ds


def socp_solver(cost, A, b, c, d, x_init=None):
    """socp optimization"""
    # optimization variable [u d f].T
    m = len(A)
    X = cp.Variable(3)
    if x_init is not None:
        X.value = x_init
    soc_constraints = [
        cp.SOC(c[i] @ X + d[i], A[i] @ X + b[i].squeeze()) for i in range(m)
    ]
    prob = cp.Problem(cp.Minimize(cost @ X), soc_constraints)
    if x_init is not None:
        prob.solve(warm_start=True, verbose=True)
    else:
        prob.solve()
    if 'infeasible' in prob.status:
        return None, prob
    return X.value[0], X.value[1], prob


def filter_cost(u, gp_model, z, v_des):
    """ Computes the error between the predicted mean the nominal input """
    m = gp_model.output_dim
    u = u.reshape(m, 1)

    query = np.hstack((z.T, u.T))

    # query = np.reshape(np.array([z[0,0], u[0]]),(1,2))
    v_query = gp_model.predict(query)
    mean = v_query[0].numpy()
    sigma2 = v_query[1].numpy()

    cost = (u + mean - v_des).T @ (u + mean - v_des) + sigma2
    return cost[0]  # + sigma2


def filter_constraint(u, gp_model, z, v_des, w, beta, V1):
    """ Computes the probabilistic worst case derivative of the chosen lyapunov safety function"""
    m = gp_model.output_dim

    u = u.reshape(m, 1)
    query = np.hstack((z.T, u.T))
    v_query = gp_model.predict(query)
    mean = v_query[0].numpy()
    sigma2 = v_query[1].numpy()
    V = w.dot(v_des) - (w.dot(mean) + beta * np.absolute(w) * np.sqrt(sigma2))

    return 2.0 * V[0] - V1[0]
    # return -1000


def safety_filter(gp_model, z, v_des, w, beta, u0, V1, eps):
    if np.linalg.norm(w) > eps:
        con = {'type': 'ineq', 'fun': filter_constraint, 'args': [gp_model, z, v_des, w[0], beta, V1]}
        # res = minimize(filter_cost, x0=u0, constraints=[con], args=(gp_model, z, v_des),
        #                method='trust-constr', options={'disp': False})
        res = minimize(filter_cost, x0=u0, args=(gp_model, z, v_des),
                       method='trust-constr', options={'disp': False})
        u = res.x

        if res.constr_violation > 1.0E-7:
            success = False
        else:
            success = True

    else:
        res = minimize(filter_cost, x0=u0, args=(gp_model, z, v_des), options={'disp': False})
        u = res.x

        success = True

    # if not(res.success):
    #    print("Success: %s" % res.success)
    #    print(res.message)
    return u, success


def unconstrained_closed_form_soln(gp_model, z, v_des):
    query = np.hstack((z.T, np.zeros((1, 1))))
    query = torch.from_numpy(query)
    gamma1, gamma2, gamma3, gamma4, gamma5 = gp_model.model.compute_gammas(query)
    v_des = torch.from_numpy(v_des).double()
    # u = (2*gamma1*v_des - 2*gamma2*gamma3)/(gamma1**2 + gamma3**2)
    u = -(gamma1 * gamma2 - gamma2 * v_des + 0.5 * gamma4) / (gamma2 ** 2 + gamma5)
    return u.numpy()


def unconstrained_closed_form_soln_for_difference_gp(gp_model, z, v_des, quad):
    query = np.hstack((z.T, np.zeros((1, 1))))
    query = torch.from_numpy(query)
    gamma1, gamma2, gamma3, gamma4, gamma5 = gp_model.model.compute_gammas(query)
    gamma1 = gamma1 + torch.from_numpy(quad.beta(z)).double()
    gamma2 = gamma2 + torch.from_numpy(quad.alpha(z)).double()
    v_des = torch.from_numpy(v_des).double()
    u = (-(gamma1 * gamma2) + gamma2 * v_des - 0.5 * gamma4) / (gamma2 ** 2 + gamma5)
    # u = -(gamma1*(1+gamma2) - (1+gamma2)*v_des+0.5*gamma4)/((1+gamma2)**2 + gamma5)
    # u = (-gamma1  + v_des) / (1 + gamma2)
    # u = (-(gamma1 + torch.from_numpy(quad.beta(z)).double()) + v_des) / (gamma2 + torch.from_numpy(quad.alpha(z)).double())
    return u.numpy()
