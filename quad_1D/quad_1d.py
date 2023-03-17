import numpy as np
import casadi as cs
from control import lqr
from scipy.linalg import expm
from utils.math_utils import rk_discrete, euler_discrete, discretize_linear_system

class Quad1D:
    def __init__(self, thrust, tau, gamma, dt, ref_type='increasing_sine'):
        self.T = thrust
        self.tau = tau
        self.gamma = gamma
        self.dt = dt
        self.ref_type = ref_type

        self.n = 3
        self.m = 1
        self.state_names = ['z0', 'z1', 'z2']
        self.state_indices = {'z0': 0,
                              'z1': 1,
                              'z2': 2}

        self.A = np.diag(np.ones(self.n-1), k=1)
        self.B = np.zeros((self.n,self.m))
        self.B[-1,0] = 1

        M = np.zeros((self.m + self.n, self.m + self.n))
        M[0:self.n, 0:self.n] = self.A * self.dt
        M[0:self.n, self.n:self.m + self.n] = self.B * self.dt
        expM = expm(M)
        self.Ad = expM[0:self.n, 0:self.n]
        self.Bd = expM[0:self.n, self.n:]

        self.cs_alpha = self.make_cs_alpha()
        self.cs_beta = self.make_cs_beta()
        self.cs_v_from_u = self.make_cs_v_from_u()
        self.cs_u_from_v = self.make_cs_u_from_v()
        self.cs_true_flat_dynamics_from_v = self.make_cs_true_flat_dynamics_from_v()
        self.cs_true_flat_dynamics = self.make_cs_true_flat_dynamics()
        self.cs_nonlin_dyn, self.cs_nonlin_jac = self.make_cs_nonlinear_dyn()
        self.cs_nonlin_dyn_discrete = self.make_cs_nonlinear_dyn_discrete()
        self.cs_lin_dyn, self.Ar, self.Br = self.make_real_lin_dyn()
        self.cs_x_from_z = self.make_x_from_z()
        self.cs_z_from_x = self.make_z_from_x()
        self.cs_true_flat_dyn_from_x_and_u = self.make_cs_true_flat_dyn_from_x_and_u()


        self.Q = None
        self.P = None
        self.S = None
        self.K_gain = None
        self.c3 = None

    def make_cs_nonlinear_dyn(self):
        x = cs.SX.sym('x')
        u = cs.SX.sym('u')
        x_dot = cs.SX.sym('x_dot')
        theta = cs.SX.sym('theta')

        X = cs.vertcat(x,x_dot,theta)
        X_dot = cs.vertcat( x_dot,
                            self.T*cs.sin(theta) - self.gamma*x_dot,
                            1/self.tau*(u - theta))
        nonlin_dyn = cs.Function('nonlin_dyn',
                                 [X,u],
                                 [X_dot],
                                 ['x', 'u'],
                                 ['x_dot'])

        dfdx = cs.jacobian(X_dot, X)
        dfdu = cs.jacobian(X_dot, u)
        df_func = cs.Function('df',
                              [X, u],
                              [dfdx, dfdu],
                              ['x', 'u'],
                              ['dfdx', 'dfdu'])

        return nonlin_dyn, df_func

    def make_real_lin_dyn(self):
        x_eq = np.zeros((3,1))
        u_eq = 0.0

        dfdxdfdu = self.cs_nonlin_jac(x=x_eq, u=u_eq)
        dfdx = dfdxdfdu['dfdx'].toarray()
        dfdu = dfdxdfdu['dfdu'].toarray()


        Ar, Br = discretize_linear_system(dfdx, dfdu, self.dt, exact=True)

        x = cs.SX.sym('x', 3)
        u = cs.SX.sym('u', 1)
        x_dot = Ar @ x + Br @ u # Don't need Delta because eq is 0.
        lin_dyn = cs.Function('lin_dyn',
                              [x, u],
                              [x_dot],
                              ['x0', 'p'],
                              ['xf'])
        return lin_dyn, Ar, Br


    def make_cs_nonlinear_dyn_discrete(self):
        nonlin_dyn_discrete = rk_discrete(self.cs_nonlin_dyn, self.n, self.m, self.dt)
        #nonlin_dyn_discrete = euler_discrete(self.cs_nonlin_dyn, self.n, self.m, self.dt)
        return nonlin_dyn_discrete

    def alpha(self, z):
        alpha = self.T/self.tau*np.sqrt((1-((z[2]+self.gamma*z[1])/self.T)**2))
        return alpha

    def make_cs_alpha(self):
        z = cs.SX.sym('z', self.n)
        alpha = self.T/self.tau*cs.sqrt((1-((z[2]+self.gamma*z[1])/self.T)**2))
        func = cs.Function('alpha',
                           [z],
                           [alpha],
                           ['z'],
                           ['alpha'])
        return func

    def beta(self, z):
        beta = - self.T/self.tau*np.sqrt((1-((z[2]+self.gamma*z[1])/self.T)**2))*np.arcsin((z[2]+self.gamma*z[1])/self.T) \
               -self.gamma*(z[2]+self.gamma*z[1]) + self.gamma**2*z[2]
        return beta

    def make_cs_beta(self):
        z = cs.SX.sym('z', self.n)
        beta = - self.T/self.tau*cs.sqrt((1-((z[2]+self.gamma*z[1])/self.T)**2))*cs.arcsin((z[2]+self.gamma*z[1])/self.T) \
               -self.gamma*(z[2]+self.gamma*z[1]) + self.gamma**2*z[2]
        func = cs.Function('beta',
                           [z],
                           [beta],
                           ['z'],
                           ['beta'])
        return func

    def v_from_u(self,u, z):
        v = self.alpha(z)*u + self.beta(z)
        return v

    def make_cs_v_from_u(self):
        z = cs.SX.sym('z', self.n)
        u = cs.SX.sym('u', self.m)

        v = self.cs_alpha(z=z)['alpha']@u + self.cs_beta(z=z)['beta']
        func = cs.Function('v_from_u',
                           [z, u],
                           [v],
                           ['z', 'u'],
                           ['v'])
        return func


    def u_from_v(self, v, z):
        u = (v-self.beta(z))/self.alpha(z)
        return u

    def make_cs_u_from_v(self):
        z = cs.SX.sym('z', self.n)
        v = cs.SX.sym('v', self.m)

        u = (v-self.cs_beta(z=z)['beta'])/self.cs_alpha(z=z)['alpha']
        func = cs.Function('u_from_v',
                           [z, v],
                           [u],
                           ['z', 'v'],
                           ['u'])
        return func

    def true_flat_dynamics_from_v(self, z, v):
        z_next = self.Ad@z + self.Bd*v
        return z_next

    def make_cs_true_flat_dynamics_from_v(self):
        z = cs.SX.sym('z', self.n)
        v = cs.SX.sym('v', self.m)

        z_next = self.Ad@z + self.Bd*v
        func = cs.Function('z_next',
                           [z, v],
                           [z_next],
                           ['z', 'v'],
                           ['z_next'])
        return func

    def true_flat_dynamics(self, z, u):
        v = self.v_from_u(u, z)
        z_next = self.true_flat_dynamics_from_v(z, v)
        return z_next, v

    def make_cs_true_flat_dynamics(self):
        z = cs.SX.sym('z', self.n)
        u = cs.SX.sym('u', self.m)


        v = self.cs_v_from_u(z=z, u=u)['v']
        z_next = self.cs_true_flat_dynamics_from_v(z=z, v=v)['z_next']

        func = cs.Function('z_next',
                           [z, u],
                           [z_next, v],
                           ['z', 'u'],
                           ['z_next', 'v'])
        return func

    def make_cs_true_flat_dyn_from_x_and_u(self):
        x = cs.SX.sym('x', self.n)
        u = cs.SX.sym('u', self.m)
        z = self.cs_z_from_x(x=x)['z']
        z_next = self.cs_true_flat_dynamics(z=z, u=u)['z_next']
        x_next = self.cs_x_from_z(z=z_next)['x']

        func = cs.Function('x_next',
                           [x, u],
                           [x_next],
                           ['x', 'u'],
                           ['x_next'])
        return func


    def make_x_from_z(self):
        z = cs.SX.sym('z', self.n)

        x = cs.vertcat(z[0],
                       z[1],
                       cs.arcsin( (z[2] + self.gamma*z[1])/self.T))
        x_from_z = cs.Function('x_from_z',
                               [z],
                               [x],
                               ['z'],
                               ['x'])
        return x_from_z

    def make_z_from_x(self):
        x = cs.SX.sym('x', self.n)

        z = cs.vertcat(x[0],
                       x[1],
                       self.T*cs.sin(x[2]) - self.gamma*x[1])
        z_from_x = cs.Function('z_from_x',
                               [x],
                               [z],
                               ['x'],
                               ['z'])
        return z_from_x

    def reference_generator(self, t, Amp, omega, ref_type=None):
        """
        Increasing sign such that quad flies in increasing 2D spiral
        Same oscillation in both x and z
        """
        if ref_type is None:
            ref_type = self.ref_type

        # z_ref = np.zeros((8,1))
        if type(t) == np.ndarray:
            n = t.shape[0]
            z_ref = np.zeros((self.n, n))
            v_ref = np.zeros((self.m, n))
        else:
            z_ref = np.zeros((self.n, 1))
            v_ref = np.zeros((self.m, 1))

        if ref_type == 'increasing_sine':
            z_ref[0] = Amp * t * np.sin(omega * t)
            z_ref[1] = Amp * np.sin(omega * t) + Amp * t * omega * np.cos(omega * t)
            z_ref[2] = 2 * omega * Amp * np.cos(omega * t) - Amp * t * omega ** 2 * np.sin(omega * t)
            v_ref[0] = -3 * omega ** 2 * Amp * np.sin(omega * t) - Amp * t * omega ** 3 * np.cos(omega * t)
        elif ref_type == 'increasing_freq':
            omega = omega*np.exp(0.05*t)
            z_ref[0] = Amp * np.sin(omega * t)
            z_ref[1] = Amp * omega * np.cos(omega * t)
            z_ref[2] =  - Amp * omega ** 2 * np.sin(omega * t)
            v_ref[0] =  - Amp * omega ** 3 * np.cos(omega * t)
        elif ref_type == 'step':
            delta_t = 3.0
            #z_ref[0] = 0.5 + 0.5*np.tanh((t-delta_t)*omega)
            #z_ref[1] = 0.5*Amp*omega*(1-np.tanh(omega*(t-delta_t))**2)
            #z_ref[2] = -Amp*omega**2*(1-np.tanh(omega*(t-delta_t))**2)*np.tanh(omega*(t-delta_t))
            #v_ref[0] = -Amp*omega**3*(1-np.tanh(omega*(t-delta_t))**2)**2 + 2.0*Amp*omega**3*(1-np.tanh(omega*(t-delta_t))**2)*np.tanh(omega*(t-delta_t))
            z_ref[0] = Amp*np.heaviside(t-delta_t,1.0)
            z_ref[1] = 0.0
            z_ref[2] = 0.0
            v_ref[0] = 0.0

        else:
            raise ValueError('Reference type not implemented.')

        return z_ref, v_ref

    def real_reference_generator(self, t, Amp, omega):
        z_ref, v_ref = self.reference_generator(t, Amp, omega)
        x_ref = self.cs_x_from_z(z=z_ref)['x'].toarray()
        u_ref = self.cs_u_from_v(z=z_ref, v=v_ref)['u'].toarray()
        return x_ref, u_ref

    def lqr_gain_and_ARE_soln(self, Q, R):
        K_gain, P, S = lqr(self.A, self.B, Q, R)
        S = Q + K_gain.T.dot(R).dot(K_gain)
        K_gain = np.asarray(K_gain)
        S = np.asarray(S)
        P = np.asarray(P)
        self.Q = Q
        self.P = P
        self.S = S
        self.K_gain = K_gain
        self.c3 = np.min(np.linalg.eig(self.S)[0]) / np.max(np.linalg.eig(self.P)[0])
