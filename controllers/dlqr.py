import numpy as np
import scipy as sp
import casadi as cs
import control

from copy import deepcopy
from utils.math_utils import get_cost_weight_matrix, csQuadCost

class DLQR:
    def __init__(self,
                 quad,
                 dt: float = 0.1,
                 q_lqr: list = [1],
                 r_lqr: list = [1],
                 reference_generator = None,
                 con_tol: float = 0.0,
                 bound = None,
                 **kwargs
                 ):


        # setup env
        self.quad = quad
        self.dt = dt
        self.ny = self.quad.n
        self.nu = self.quad.m
        self.nz = self.quad.n
        self.bound = bound
        self.name = 'DLQR'

        if reference_generator is None:
            self.reference_generator = quad.reference_generator
        else:
            self.reference_generator = reference_generator

        # Setup controller parameters
        self.Q = get_cost_weight_matrix(q_lqr, self.nz)
        self.R = get_cost_weight_matrix(r_lqr, self.nu)

        self.Ad = quad.Ad
        self.Bd = quad.Bd

        # Compute Gain.
        self.K, self.P = self.compute_gain()

    def compute_gain(self):
        K, P, E = control.dlqr(self.Ad, self.Bd, self.Q, self.R)
        return K, P

    def select_flat_input(self, z, t, params):
        z_ref, v_ref = self.reference_generator(t, params['Amp'], params['omega'])
        v_des = - self.K @ (z - z_ref) + v_ref
        zd = z
        return_status = True
        return zd, v_des, return_status

    def compute_feedback_input(self,
                           gp,
                           z,
                           z_ref,
                           v_ref,
                           x_init=None,
                           t=None,
                           params=None,
                           **kwargs):
        _, v_des, _ = self.select_flat_input(z, t, params)
        u = self.quad.u_from_v(v_des, z)
        if self.bound is not None:
            u = np.clip(u, -self.bound, self.bound)
        return u, v_des, True, 0
