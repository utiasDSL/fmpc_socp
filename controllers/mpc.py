"""Model Predictive Control.

"""
import numpy as np
import casadi as cs

from sys import platform
from copy import deepcopy

#from safe_control_gym.controllers.base_controller import BaseController
#from safe_control_gym.controllers.mpc.mpc_utils import get_cost_weight_matrix, compute_discrete_lqr_gain_from_cont_linear_system, rk_discrete, compute_state_rmse, reset_constraints
#from safe_control_gym.envs.benchmark_env import Task
#from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS, create_constraint_list

from copy import deepcopy
from utils.math_utils import get_cost_weight_matrix, csQuadCost, rk_discrete

class MPC:
    """MPC with full nonlinear model.

    """

    def __init__(
            self,
            quad,
            name='MPC',
            horizon: int = 10,
            dt: float = 0.1,
            q_mpc: list = [1],
            r_mpc: list = [1],
            solver: str = 'ipopt',
            constraints: dict = None,
            dynamics: cs.Function = None,
            reference_generator = None,
            upper_bounds: dict = None,
            lower_bounds: dict = None,
            input_bound: float = None,
            con_tol: float = 0.0,
            ):
        """Creates task and controller.

        Args:
            env_func (Callable): function to instantiate task/environment.
            horizon (int): mpc planning horizon.
            q_mpc (list): diagonals of state cost weight.
            r_mpc (list): diagonals of input/action cost weight.
            warmstart (bool): if to initialize from previous iteration.
            soft_constraints (bool): Formulate the constraints as soft constraints.
            terminate_run_on_done (bool): Terminate the run when the environment returns done or not.
            constraint_tol (float): Tolerance to add the the constraint as sometimes solvers are not exact.
            output_dir (str): output directory to write logs and results.
            additional_constraints (list): List of additional constraints
            use_gpu (bool): False (use cpu) True (use cuda).
            seed (int): random seed.

        """
        # setup env
        self.quad = quad
        self.solver = solver
        self.dt = dt
        self.T = horizon
        self.nx = self.quad.n
        self.nu = self.quad.m
        self.nz = self.quad.n
        self.name = name

        self.constraints = {}
        self.input_bound = input_bound
        # Handle constraints
        if upper_bounds is not None and lower_bounds is not None:
            assert len(upper_bounds.keys()) == len(upper_bounds), 'Provide upper and lower bounds together.'
            for state in self.quad.state_names:
                if state in upper_bounds.keys():
                    self.constraints[state] = {}
                    self.constraints[state]['upper_bound'] = upper_bounds[state]
                    self.constraints[state]['lower_bound'] = lower_bounds[state]
            self.con_tol = con_tol

        # Setup controller parameters
        self.Q = get_cost_weight_matrix(q_mpc, self.nx)
        self.R = get_cost_weight_matrix(r_mpc, self.nu)

        # compute mappings and flat dynamics
        if dynamics is None:
            #self.dyn_func = quad.cs_nonlin_dyn_discrete
            self.dyn_func = quad.cs_lin_dyn
        else:
            self.dyn_func = dynamics

        if reference_generator is None:
            self.reference_generator = quad.real_reference_generator
        else:
            self.reference_generator = reference_generator

        self.v_hist = None

        # setup optimizer
        self.x_prev = None
        self.u_prev = None
        self.setup_optimizer()

    def setup_optimizer(self):
        nz, nu = self.nz, self.nu
        T = self.T
        # Define optimizer and variables.
        opti = cs.Opti()
        x_var = opti.variable(nz, T+1)
        u_var = opti.variable(nu, T)
        x_0 = opti.parameter(nz, 1)
        x_ref = opti.parameter(self.nz, T+1)

        # Dynamics constraints
        opti.subject_to(x_var[:,0] == x_0)
        for i in range(T):
            next_state = self.dyn_func(x0=x_var[:,i], p=u_var[:,i])['xf']
            opti.subject_to(x_var[:, i+1] == next_state)

        cost = 0
        for i in range(1,T+1):
            #cost += csQuadCost(x_var[[0,4],i], x_ref[[0,4],i], self.Q)
            cost += csQuadCost(x_var[:, i], x_ref[:, i], self.Q)
        for i in range(T):
            cost += csQuadCost(u_var[:,i], np.zeros((nu,1)), self.R)

        if not(self.constraints == {}):
            for state in self.constraints.keys():
                for i in range(0,T+1):
                    opti.subject_to(opti.bounded(self.constraints[state]['lower_bound'] + self.con_tol,
                                                 x_var[self.quad.state_indices[state], i],
                                                 self.constraints[state]['upper_bound'] - self.con_tol))
        if self.input_bound is not None:
            for i in range(0,T):
                opti.subject_to(opti.bounded(-self.input_bound, u_var[:,i], self.input_bound))

        opti.minimize(cost)
        opts = {"expand": True}
        if platform == "linux":
            opts.update({"print_time": 1})
            opti.solver(self.solver, opts)
        elif platform == "darwin":
            opts.update({"ipopt.max_iter": 100})
            opti.solver('ipopt', opts)
        else:
            print("[ERROR]: CasADi solver tested on Linux and OSX only.")
            exit()
        #opts.update({"qpsol_options": {"tol": 1e-4}})
        opti.solver(self.solver, opts)

        self.opti_dict = {
            "opti": opti,
            "x_var": x_var,
            "u_var": u_var,
            "x_0": x_0,
            "x_ref": x_ref,
            "cost": cost
        }

    def compute_feedback_input(self,
                               gp,
                               z,
                               x_ref,
                               v_ref,
                               x_init=None,
                               t=None,
                               params=None,
                               **kwargs):

        x_0 = self.quad.cs_x_from_z(z=z)['x'].toarray()
        if t is None:
            raise ValueError("MPC needs the current sim time to genereate the reference.")
        if params is None:
            raise ValueError("MPC needs Ampa and Omega to genereate the reference.")
        x, u, return_status = self.select_flat_input(x_0, t, params)
        z_opt = self.quad.cs_z_from_x(x=x)['z']
        v_des = self.quad.cs_v_from_u(z=z_opt, u=u)['v'].toarray()

        return u, v_des, return_status, 0

    def select_flat_input(self, obs, t, params):
        amp = params["Amp"]
        omega = params["omega"]
        #x_obs = obs[:,None]
        x_obs = obs
        if self.x_prev is not None:
            x_error = self.x_prev[:, 1] - np.squeeze(x_obs)
            self.results_dict['x_error'].append(deepcopy(x_error))
        self.results_dict['flat_states'].append(deepcopy(x_obs))
        opti_dict = self.opti_dict
        opti = opti_dict["opti"]
        x_var = opti_dict["x_var"]
        u_var = opti_dict["u_var"]
        x_0 = opti_dict["x_0"]
        x_ref = opti_dict["x_ref"]
        cost = opti_dict["cost"]
        # Assign the initial state.
        opti.set_value(x_0, x_obs)
        # Assign reference trajectory within horizon.
        t_ref = np.linspace(t, t+self.dt*self.T, num=self.T+1)
        goal_states = self.reference_generator(t_ref, amp, omega)[0]
        self.results_dict['goal_states'].append(goal_states)
        opti.set_value(x_ref, goal_states)
        if self.x_prev is not None and self.u_prev is not None:
            x_guess = np.hstack((x_obs, self.x_prev[:,1:]))
            u_guess = np.hstack((self.u_prev[:,1:], self.u_prev[:,-1,None]))
            opti.set_initial(x_var, x_guess)
            opti.set_initial(u_var, u_guess)
        # Solve the optimization problem.
        try:
            sol = opti.solve()
            x_val, u_val = np.atleast_2d(sol.value(x_var)), np.atleast_2d(sol.value(u_var))
            self.x_prev = x_val
            self.u_prev = u_val
            self.results_dict['horizon_states'].append(deepcopy(self.x_prev))
            self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev))
            return_status = True
        except RuntimeError as e:
            print(e)
            return_status = False #opti.return_status()
            print("[Warn]: %s" % return_status)
            #raise ValueError()

        return x_val[:,0], u_val[:,0], return_status


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
                              'x_error': [],
                              'augmented_states': [],
                              'frames': [],
                              'state_mse': [],
                              'common_cost': [],
                              'state': [],
                              'state_error': [],
                              't_wall': []
                              }

    def reset(self, dynamics=None):
        self.x_prev = None
        self.u_prev = None

        # Reset Data Storage.
        self.setup_results_dict()

        # compute mappings and flat dynamics
        if dynamics is None:
            if self.dyn_func is None:
                self.dyn_func = self.quad.cs_nonlin_dyn_discrete
        else:
            self.dyn_func = dynamics


        # setup optimizer
        self.setup_optimizer()


