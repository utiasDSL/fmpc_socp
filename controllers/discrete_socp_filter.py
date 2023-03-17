import numpy as np
from numpy.linalg import norm
import cvxpy as cp
import torch

class DiscreteSOCPFilter:
    def __init__(self, name, quad, beta, d_weight=25.0, input_bound=None, state_bound=None, ctrl=None, gp=None):
        self.name = name
        self.quad = quad
        self.beta = beta
        self.d_weight = d_weight
        self.gp = gp
        if ctrl.P is None:
            raise ValueError('Controller must have gain matrics P, Q, R, K')
        else:
            self.P = ctrl.P
            self.R = ctrl.R
            self.Q = ctrl.Q
            self.K = ctrl.K
            self.Ad = ctrl.Ad
            self.Bd = ctrl.Bd
        self.input_bound = input_bound
        self.state_bound = state_bound
        if ctrl is not None:
            if not(ctrl.name in ['FMPC', 'DLQR']):
                raise ValueError('Only FMPC or DLQR can be used here for now.')
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
            h = state_bound['h']
            bcon = state_bound['b']
            phi_p = state_bound['phi_p']
            self.del_sig = phi_p * np.sqrt(h.T @ self.Bd @ self.Bd.T @ h)
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

    def compute_feedback_input(self, gp, z, z_ref, v_ref, x_init=None, t=None, params=None, **kwargs):
        """ Compute u so it can be used in feedback function"""
        if gp is None:
            gp = self.gp
        if self.ctrl is None:
            v_des = - self.K.dot((z - z_ref)) + v_ref
            u, d_sf = self.solve(gp, z, z_ref, v_des, x_init=x_init)
        else:
            zd, v_des, return_status = self.ctrl.select_flat_input(z, t, params)
            zd = np.atleast_2d(zd)
            u, d_sf = self.solve(gp, zd, z_ref, v_des, x_init=x_init)
        if 'optimal' in self.prob.status:
            success = True
        else:
            success = False
        return u, v_des, success, d_sf

    def solve(self, gp_model, z, z_ref, v_des, x_init=np.zeros((3,))):
        e_k = z - z_ref
        # Compute state dependent values
        gam1, gam2, gam3, gam4, gam5 = get_gammas(z, gp_model)
        w, norm_w = compute_w(e_k, self.Ad, self.Bd, self.P, self.K)
        #v_nom = -self.K @ e_k + v_des
        v_nom = v_des

        # Compute cost coefficients
        cost = compute_cost(gam1, gam2, gam4, v_des)
        self.cost.value = cost

        # Compute stablity filter coeffs
        A1, b1, c1, d1 = stab_filter_matrices(gam1, gam2, gam3, gam4, gam5,
                                              self.Q, self.R, self.P, self.K, self.Bd, e_k,
                                              norm_w, w,
                                              self.input_bound, v_nom, self.beta)
        self.A1.value = A1
        self.b1.value = b1
        self.c1.value = c1
        self.d1.value = d1

        # Compute dummy var mats
        A2, b2, c2, d2 = dummy_var_matrices(gam2, gam5, self.d_weight)
        self.A2.value = A2
        self.b2.value = b2
        self.c2.value = c2
        self.d2.value = d2

        # Compute state constraints.

        if self.state_bound is not None:

            Astate, bstate, cstate, dstate = state_con_matrices(z, gam1, gam2, gam3, gam4, gam5,
                                                                self.state_bound, self.Ad, self.Bd, self.del_sig,
                                                                self.d_weight)

            self.Astate.value = Astate
            self.bstate.value = bstate.squeeze()
            self.cstate.value = cstate
            self.dstate.value = dstate.squeeze()

        self.X.value = x_init
        self.prob.solve(solver='MOSEK', warm_start=True, verbose=True) # SCS was used in paper
        if 'optimal' in self.prob.status:
            return self.X.value[0], self.X.value[1]
        else:
            return 0, 0

def get_gammas(z, gp_model):
    query_np = np.hstack((z.T, np.zeros((1, 1)))) # ToDo: Why is this 0 added here?
    query = torch.from_numpy(query_np).double()
    gamma1, gamma2, gamma3, gamma4, gamma5 = gp_model.model.compute_gammas(query)
    gamma1 = gamma1.numpy().squeeze()
    gamma2 = gamma2.numpy().squeeze()
    gamma3 = gamma3.numpy().squeeze()
    gamma4 = gamma4.numpy().squeeze()
    gamma5 = gamma5.numpy().squeeze()
    return gamma1, gamma2, gamma3, gamma4, gamma5

def compute_cost(gam1, gam2, gam4, v_des):
    cost = np.array([[2 * gam1 * gam2 - 2 * gam2 * v_des.squeeze() + gam4, 0, 1]])
    return cost
def compute_w(e_k, Ad, Bd, P, K):
    w = e_k.T @ (Ad - Bd @ K).T @ P @ Bd
    return w.squeeze(), np.linalg.norm(w)

def stab_filter_matrices(gam1,
                         gam2,
                         gam3,
                         gam4,
                         gam5,
                         Q, R, P, K, Bd,
                         e_k,
                         norm_w, w,
                         u_max, v_nom, beta):
    A1, b1 = stab_filter_A1_and_b1(gam3, gam4, gam5, norm_w)
    c1, d1 = stab_filter_c1_and_d1(gam1, gam2,
                          Q, R, P, K, Bd,
                          e_k, w,
                          u_max, v_nom, beta)
    return A1, b1, c1, d1

def stab_filter_A1_and_b1(gam3,
                          gam4,
                          gam5,
                          norm_w):
    A1 = np.array([[norm_w*np.sqrt(gam5), 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    b1 = np.array([[norm_w*gam4 / (2 * np.sqrt(gam5))],
                   [norm_w*np.sqrt(gam3 - 0.25 * gam4 ** 2 / gam5)],
                   [0]])
    return A1, b1.squeeze()

def stab_filter_c1_and_d1(gam1,
                          gam2,
                          Q, R, P, K, Bd,
                          e_k,
                          w,
                          u_max, v_nom, beta):
    d_a = e_k.T @ P @ e_k
    d_b = e_k.T @ (P - Q - K.T @ R @ K) @ e_k
    d_c = np.max([(gam1 + gam2*u_max - v_nom)**2, (gam1 + gam2*(-u_max) - v_nom)**2])* Bd.T @ P @ Bd
    #d_c = 0.0
    d_d = 2*w*(gam1 - v_nom)
    d1 = (-1/(2*beta))*(d_a - d_b - d_c + d_d)

    c1 = (-1/(2*beta))*np.array([[2*w*gam2, 1.0, 0.0]])

    return c1, d1.squeeze()

def dummy_var_matrices(gam2, gam5, d_weight):
    A2 = np.array([[2.0 * np.sqrt(gam2 ** 2 + gam5), 0, 0],
                   [0, 2.0 * d_weight, 0],
                   [0, 0, -1.0]])
    b2 = np.array([[0], [0], [1.0]])
    c2 = np.array([[0, 0, 1.0]])
    d2 = 1

    return A2, b2.squeeze(), c2, d2

def state_con_matrices(z, gam1, gam2, gam3, gam4, gam5,
                       state_bound, Ad, Bd, del_sig, d_weight):
    h = state_bound['h']
    bcon = state_bound['b']
    Astate = np.array([[float(del_sig*np.sqrt(gam5)), 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])
    bstate = np.array([[float(del_sig*gam4 / (2 * np.sqrt(gam5)))],
                       [float(del_sig*np.sqrt(gam3 - 0.25 * gam4 ** 2 / gam5))],
                       [0]])
    cstate = np.array([[float(-h.T @ Bd * gam2), d_weight, 0.0]])
    dstate = -h.T @ Ad @ z - h.T @ Bd * gam1 + bcon
    return Astate, bstate, cstate, dstate
