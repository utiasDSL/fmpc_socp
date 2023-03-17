"""Flat MPC based on Greef 2018 'Flatness-based Model Predictive Control for Quadrotor Trajectory Tracking'

"""

import numpy as np
import scipy as sp
import control
import casadi as cs
from copy import deepcopy
from utils.math_utils import get_cost_weight_matrix, csQuadCost

class FMPC:
    def __init__(self,
                 quad,
                 horizon: int = 10,
                 dt: float = 0.1,
                 q_mpc: list = [1],
                 r_mpc: list = [1],
                 solver: str = 'qrqp',
                 reference_generator = None,
                 constraints: dict = None,
                 upper_bounds: dict = None,
                 lower_bounds: dict = None,
                 con_tol: float = 0.0,
                 **kwargs
                 ):


        # setup env
        self.quad = quad
        self.solver = solver
        self.dt = dt
        self.T = horizon
        self.ny = self.quad.n
        self.nu = self.quad.m
        self.nz = self.quad.n
        self.name = 'FMPC'

        self.constraints = {}
        # Handle constraints
        if upper_bounds is not None and lower_bounds is not None:
            assert len(upper_bounds.keys()) == len(upper_bounds), 'Provide upper and lower bounds together.'
            for state in self.quad.state_names:
                if state in upper_bounds.keys():
                    self.constraints[state] = {}
                    self.constraints[state]['upper_bound'] = upper_bounds[state]
                    self.constraints[state]['lower_bound'] = lower_bounds[state]
            self.con_tol = con_tol

        if reference_generator is None:
            self.reference_generator = quad.real_reference_generator
        else:
            self.reference_generator = reference_generator

        # Setup controller parameters
        self.Q = get_cost_weight_matrix(q_mpc, self.nz)
        self.R = get_cost_weight_matrix(r_mpc, self.nu)

        # compute mappings and flat dynamics
        self.flat_dyn_func = self.quad.cs_true_flat_dynamics_from_v
        self.Ad = self.quad.Ad
        self.Bd = self.quad.Bd
        self.v_hist = None

        # setup optimizer
        self.z_prev = None
        self.v_prev = None
        self.setup_optimizer()
        self.P, self.K = self.compute_K_and_P()

    def compute_K_and_P(self):
        # Equations taken from Borrelli Sec 8.3 but with the opposite sign for K as we are using the convention
        # u = -Kx and they use u = Kx
        P = deepcopy(self.Q)*100.0
        for i in range(self.T):
            P = self.Ad.T @ P @ self.Ad + self.Q - self.Ad.T @ P @ self.Bd @ np.linalg.pinv(self.Bd.T @ P @ self.Bd + self.R) @ self.Bd.T @ P @ self.Ad
        K = np.linalg.pinv(self.Bd.T @ P @ self.Bd + self.R) @ self.Bd.T @ P @ self.Ad
        #K, P, _ = control.dlqr(self.Ad, self.Bd, P, self.R)
        return P, K

    def setup_optimizer(self):
        nz, nu = self.nz, self.nu
        T = self.T
        # Define optimizer and variables.
        if self.solver in ['qrqp', 'qpoases']:
            opti = cs.Opti('conic')
        else:
            opti = cs.Opti()
        z_var = opti.variable(nz, T+1)
        v_var = opti.variable(nu, T)
        z_0 = opti.parameter(nz, 1)
        z_ref = opti.parameter(self.nz, T+1)

        # Dynamics constraints
        opti.subject_to(z_var[:,0] == z_0)
        for i in range(T):
            next_state = self.flat_dyn_func(z=z_var[:,i], v=v_var[:,i])['z_next']
            opti.subject_to(z_var[:, i+1] == next_state)

        cost = 0
        for i in range(1,T+1):
            #cost += csQuadCost(z_var[[0,4],i], z_ref[[0,4],i], self.Q)
            cost += csQuadCost(z_var[:, i], z_ref[:, i], self.Q)
        for i in range(T):
            cost += csQuadCost(v_var[:,i], np.zeros((nu,1)), self.R)

        if not(self.constraints == {}):
            for state in self.constraints.keys():
                for i in range(0,T+1):
                    opti.subject_to(opti.bounded(self.constraints[state]['lower_bound'] + self.con_tol,
                                                 z_var[self.quad.state_indices[state], i],
                                                 self.constraints[state]['upper_bound'] - self.con_tol))

        opti.minimize(cost)
        opts = {"expand": True}
        opts.update({"print_time": True})
        #opts.update({"qpsol_options": {"tol": 1e-4}})
        opti.solver(self.solver, opts)

        self.opti_dict = {
            "opti": opti,
            "z_var": z_var,
            "v_var": v_var,
            "z_0": z_0,
            "z_ref": z_ref,
            "cost": cost
        }

    def compute_feedback_input(self,
                               gp,
                               z,
                               z_ref,
                               v_ref,
                               x_init=None,
                               t=None,
                               params=None,
                               **kwargs):
        if t is None:
            raise ValueError("FMPC needs the current sim time to genereate the reference.")
        if params is None:
            raise ValueError("FMPC needs Ampa and Omega to genereate the reference.")
        zd, vd, return_status = self.select_flat_input(z, t, params)
        u = self.quad.cs_u_from_v(z=zd, v=vd)['u'].toarray()

        return u, vd, return_status, 0

    def select_flat_input(self, obs, t, params):
        amp = params["Amp"]
        omega = params["omega"]
        #z_obs = obs[:,None]
        z_obs = obs
        if self.z_prev is not None:
            z_error = self.z_prev[:, 1] - np.squeeze(z_obs)
            self.results_dict['z_error'].append(deepcopy(z_error))
        self.results_dict['flat_states'].append(deepcopy(z_obs))
        opti_dict = self.opti_dict
        opti = opti_dict["opti"]
        z_var = opti_dict["z_var"]
        v_var = opti_dict["v_var"]
        z_0 = opti_dict["z_0"]
        z_ref = opti_dict["z_ref"]
        cost = opti_dict["cost"]
        # Assign the initial state.
        opti.set_value(z_0, z_obs)
        # Assign reference trajectory within horizon.
        t_ref = np.linspace(t, t+self.dt*self.T, num=self.T+1)
        goal_states = self.reference_generator(t_ref, amp, omega)[0]
        self.results_dict['goal_states'].append(goal_states)
        opti.set_value(z_ref, goal_states)
        if self.z_prev is not None and self.v_prev is not None:
            z_guess = np.hstack((z_obs, self.z_prev[:,1:]))
            v_guess = np.hstack((self.v_prev[:,1:], self.v_prev[:,-1,None]))
            opti.set_initial(z_var, z_guess)
            opti.set_initial(v_var, v_guess)
        # Solve the optimization problem.
        try:
            sol = opti.solve()
            z_val, v_val = np.atleast_2d(sol.value(z_var)), np.atleast_2d(sol.value(v_var))
            self.z_prev = z_val
            self.v_prev = v_val
            self.results_dict['horizon_states'].append(deepcopy(self.z_prev))
            self.results_dict['horizon_inputs'].append(deepcopy(self.v_prev))
            return_status = True
        except RuntimeError as e:
            print(e)
            return_status = False #opti.return_status()
            print("[Warn]: %s" % return_status)
            #raise ValueError()

        return_status = opti.return_status()
            #if return_status == 'unknown':
            #    self.terminate_loop = True
            #    u_val = self.u_prev
            #    if u_val is None:
            #        print('[WARN]: MPC Infeasible first step.')
            #        u_val = np.zeros((1, self.model.nu))
            #elif return_status == 'Maximum_Iterations_Exceeded':
            #    self.terminate_loop = True
            #    u_val = opti.debug.value(u_var)
            #elif return_status == 'Search_Direction_Becomes_Too_Small':
            #    self.terminate_loop = True
            #    u_val = opti.debug.value(u_var)

        # take first one from solved action sequence
        z = z_val[:,0].reshape(self.nz, 1)
        return z, v_val[:,0], return_status
        #return z_val[:,0], v_val[:,0], return_status
        #return goal_states[:,1], v_val[:, 0]

    def setup_results_dict(self):
        """

        """
        self.results_dict = { 'obs': [],
                              'reward': [],
                              'done': [],
                              'info': [],
                              'action': [],
                              'horizon_inputs': [],
                              'horizon_states': [],
                              'goal_states': [],
                              'flat_states': [],
                              'z_error': [],
                              'augmented_states': [],
                              'frames': [],
                              'state_mse': [],
                              'common_cost': [],
                              'state': [],
                              'state_error': [],
                              't_wall': []
                              }

    def reset(self):
        self.z_prev = None
        self.v_prev = None

        # Reset Data Storage.
        self.setup_results_dict()

        # compute mappings and flat dynamics
        self.flat_dyn_func = self.quad.cs_true_flat_dynamics_from_v

        # setup optimizer
        self.setup_optimizer()

def backward_diff(u, dt):
    """ assume u is a list with u_k, u_k-1"""
    u_dot = (u[:,0,None] - u[:,1,None])/dt
    return u_dot

def circle(start_time, traj_length, sample_time, traj_period, scaling, offset):
    '''Computes the coordinates of a circle trajectory at time t.

    Args:
        t (float): The time at which we want to sample one trajectory point.
        traj_period (float): The period of the trajectory in seconds.
        scaling (float, optional): Scaling factor for the trajectory.

    Returns:
        coords_a (float): The position in the first coordinate.
        coords_b (float): The position in the second coordinate.
        coords_a_dot (float): The velocity in the first coordinate.
        coords_b_dot (float): The velocity in the second coordinate.
    '''
    times = np.arange(start_time, traj_length, sample_time)
    traj_freq = 2.0 * np.pi / traj_period
    z = scaling * np.cos(traj_freq * times) + offset[1]
    z_dot = -scaling * traj_freq * np.sin(traj_freq * times)
    z_ddot = -scaling * traj_freq**2 * np.cos(traj_freq * times)
    z_dddot = scaling * traj_freq**3 * np.sin(traj_freq * times)

    x = scaling * np.sin(traj_freq * times) + offset[0]
    x_dot = scaling * traj_freq * np.cos(traj_freq * times)
    x_ddot = -scaling * traj_freq**2 * np.sin(traj_freq * times)
    x_dddot = -scaling * traj_freq**3 * np.cos(traj_freq * times)

    ref = np.vstack((x,x_dot,x_ddot,x_dddot,z,z_dot,z_ddot,z_dddot))

    return ref

