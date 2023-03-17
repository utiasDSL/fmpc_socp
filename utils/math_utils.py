import casadi as cs
import numpy as np
import scipy

def get_cost_weight_matrix(weights,
                           dim
                           ):
    """Gets weight matrix from input args.

    """
    if len(weights) == dim:
        W = np.diag(weights)
    elif len(weights) == 1:
        W = np.diag(weights * dim)
    else:
        raise Exception("Wrong dimension for cost weights.")
    return W

def csQuadCost(y1, y2, Q):
    """
    Creates a quadratic cost term || y1 - y2 ||_Q^2 and adds it to cost for use in CasAdi optimization.
    Args:
        y1: n x 1 array
        y2: n x 1 array
        Q: n x n gain matrix
        cost: Casadi running cost
    """

    cost = (y1 - y2).T @ Q @ (y1 - y2)
    return cost

def rk_discrete(f, n, m, dt):
    """Runge Kutta discretization for the function.

    Args:
        f (casadi function): Function to discretize.
        n (int): state dimensions.
        m (int): input dimension.
        dt (float): discretization time.

    Return:
        x_next (casadi function?):
    """
    X = cs.SX.sym('X', n)
    U = cs.SX.sym('U', m)
    # Runge-Kutta 4 integration
    k1 = f(X,         U)
    k2 = f(X+dt/2*k1, U)
    k3 = f(X+dt/2*k2, U)
    k4 = f(X+dt*k3,   U)
    x_next = X + dt/6*(k1+2*k2+2*k3+k4)
    rk_dyn = cs.Function('rk_f', [X, U], [x_next], ['x0', 'p'], ['xf'])

    return rk_dyn

def euler_discrete(f, n, m, dt):
    """Euler discretization for the function.

    Args:
        f (casadi function): Function to discretize.
        n (int): state dimensions.
        m (int): input dimension.
        dt (float): discretization time.

    Return:
        x_next (casadi function?):
    """
    X = cs.SX.sym('X', n)
    U = cs.SX.sym('U', m)
    x_next = X + f(X, U)*dt
    euler_dyn = cs.Function('euler_f', [X, U], [x_next], ['x0', 'p'], ['xf'])

    return  euler_dyn

def discretize_linear_system(A,
                             B,
                             dt,
                             exact=False
                             ):
    """Discretize a linear system.

    dx/dt = A x + B u
    --> xd[k+1] = Ad xd[k] + Bd ud[k] where xd[k] = x(k*dt)

    Args:
        A: np.array, system transition matrix.
        B: np.array, input matrix.
        dt: scalar, step time interval.
        exact: bool, if to use exact discretization.

    Returns:
        Discretized matrices Ad, Bd.

    """
    state_dim, input_dim = A.shape[1], B.shape[1]
    if exact:
        M = np.zeros((state_dim + input_dim, state_dim + input_dim))
        M[:state_dim, :state_dim] = A
        M[:state_dim, state_dim:] = B
        Md = scipy.linalg.expm(M * dt)
        Ad = Md[:state_dim, :state_dim]
        Bd = Md[:state_dim, state_dim:]
    else:
        I = np.eye(state_dim)
        Ad = I + A * dt
        Bd = B * dt
    return Ad, Bd
