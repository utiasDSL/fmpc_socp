import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def plot_comparison(title, data_dicts, xkey, ykey, fig_count=0):
    fig_count += 1
    plt.figure(fig_count)
    plt.title(title)
    for data_dict in data_dicts:
        x = data_dict[xkey]
        y = data_dict[ykey]
        n_subplots = y.shape[1]
        for i in range(n_subplots):
            plt_id = int(n_subplots * 100 + 10 + i + 1)
            plt.subplot(plt_id)
            plt.plot(x, y[:,i], label=data_dict['primary_name'])
            plt.xlabel(xkey)
            plt.ylabel(ykey + '%s' % i)
    plt.legend()
    plt.show()
    return fig_count


def feedback_loop(params,
                  gp,
                  true_dynamics,
                  reference_generator,
                  primary_controller, # feedbacm function
                  secondary_controllers=None, # list of comparison controllers
                  online_learning=False,
                  fig_count=0,
                  plot=True,
                  input_bound: float=None):
    """ Run a feedback loop.

    Primary controller is used to determine the input u to be applied to the system.
    Secondary controllers contains a list of controllers whos inputs are computed
    for comparison, but not actually applied to the system.

    """
    # Parse Params
    N = params["N"] # number of time steps
    n = params["n"] # dimension of inputs
    m = params["m"] # dimension of outputs
    dt = params["dt"] # time step
    Amp = params["Amp"] # Reference traj Amplitude
    omega = params["omega"] # Reference traj freq
    if online_learning:
        n_online = params['n_online']
        learning_rate = params['learning_rate']
        variance = params['training_variance']
        n_train = params['n_train']
    # Logging
    u_log = np.zeros((N, 1))
    if secondary_controllers is not None:
        n_sec_controllers = len(secondary_controllers)
        u_secondary_log = np.zeros((N,n_sec_controllers))
        v_secondary_log = np.zeros((N, n_sec_controllers))
    z_ref_log = np.zeros((N+1,n))
    z_log = np.zeros((N+1,n))
    v_log = np.zeros((N,1))
    v_des_log = np.zeros((N,1))
    d_sf_log = np.zeros((N, 1))
    t_log = np.zeros((N+1, 1))
    error_log = np.zeros((N,1))
    solve_time_log = np.zeros((N,1))
    infeasible_index = N+1
    infeasible = False
    z, _ = reference_generator(0, Amp, omega)
    #z = np.zeros((3,1))
    u = np.zeros(1)
    x_init = np.zeros(3)
    # Main Feedback Loop
    for i in range(0, N):
        t = i * dt
        z_log[i,:] = z.T
        z_ref, v_ref = reference_generator(t, Amp, omega)
        x_init[0] = u
        t1 = time.perf_counter()
        u, v_des, success, d_sf = primary_controller.compute_feedback_input(gp,
                                                                            z,
                                                                            z_ref,
                                                                            v_ref,
                                                                            x_init=x_init,
                                                                            t=t,
                                                                            params=params)
        t2 = time.perf_counter()
        ct = t2 - t1
        print("Step: %s,  min_time: %s, success: %s,  error: %s" % (
        i, ct, success, np.sum(np.linalg.norm(z - z_ref))))
        if secondary_controllers is not None:
            # compute inputs for secondary controllers for comparisons
            for ic, sec_controller in enumerate(secondary_controllers):
                u_secondary_log[i,ic], _, _, _ = sec_controller.compute_feedback_input(gp,
                                                                                       z,
                                                                                       z_ref,
                                                                                       v_ref,
                                                                                       x_init=x_init,
                                                                                       t=t,
                                                                                       params=params)
            # compute measured flat inputs for secondary controllers for comparisons
            for ic, sec_controller in enumerate(secondary_controllers):
                _, v_secondary_log[i, ic] = true_dynamics(z, u_secondary_log[i, ic])
        if not success or any(np.isnan(np.atleast_1d(u))):
            infeasible = True
            infeasible_index = i
            print("%s INFEASIBLE with statues" % (primary_controller.name))
            break
        # step the dynamics
        if input_bound is not None:
            u = np.clip(u, -input_bound, input_bound)
        z, v_meas = true_dynamics(z, u)
        if any(np.isnan(z)):
            infeasible = True
            infeasible_index = i
            print("%s INFEASIBLE with statues" % (primary_controller.name))
            break
        # log
        t_log[i] = t
        u_log[i,:] = u
        v_log[i,:] = v_meas
        z_ref_log[i,:] = z_ref.T
        v_des_log[i] = v_des
        d_sf_log[i] = d_sf
        solve_time_log[i,:] = ct
        error_log[i,:] = np.linalg.norm(z-z_ref)
    z_log[i+1, :] = z.T
    z_ref, v_ref = reference_generator(t+dt, Amp, omega)
    t_log[i+1] = t+dt
    z_ref_log[i+1,:] = z_ref.T
    if plot:
        # Plot the states along the trajectory and compare with reference.
        fig_count += 1
        plt.figure(fig_count)
        for i in range(n):
            plt_id = int(n*100 + m*10 + i + 1)
            plt.subplot(plt_id)
            plt.plot(t_log, z_log[:, i], label=primary_controller.name)
            plt.plot(t_log, z_ref_log[:, i], label='Ref')
            plt.xlabel('t')
            plt.ylabel('z%s' % i)
            plt.title(" %s FB State z%s Comparisons" %(primary_controller.name, i))
        plt.legend()
        plt.show()
        # Plot a comparison of the flat inputs.
        fig_count += 1
        plt.figure(fig_count)
        for i in range(m):
            plt_id = int(int(m) * 100 + 10 + i + 1)
            plt.subplot(plt_id)
            plt.plot(t_log[:-1], v_des_log[:, i], label=primary_controller.name)
            if secondary_controllers is not None:
                for ic in range(n_sec_controllers):
                    plt.plot(t_log[:-1], v_secondary_log[:, ic], label=secondary_controllers[ic].name)
            plt.xlabel('t')
            plt.ylabel('v%s' % i)
        plt.title('%s FB Comparison of desired Flat Inputs' % primary_controller.name)
        plt.legend()
        plt.show()
        # Plot a comparison of the real inputs applied to the system.
        fig_count += 1
        plt.figure(fig_count)
        for i in range(m):
            plt_id = int(int(m) * 100 + 10 + i + 1)
            plt.subplot(plt_id)
            plt.plot(t_log[:-1], u_log[:, i], label=primary_controller.name)
            if secondary_controllers is not None:
                for ic in range(n_sec_controllers):
                    plt.plot(t_log[:-1], u_secondary_log[:, ic], label=secondary_controllers[ic].name)
            plt.xlabel('t')
            plt.ylabel('u%s' % i)
        plt.title('%s FB Comparison of Real inputs' % primary_controller.name)
        plt.legend()
        plt.show()
    data_dict = {}
    data_dict['params'] = params
    data_dict['t'] = t_log
    data_dict['u'] = u_log
    data_dict['v'] = v_log
    data_dict['z'] = z_log
    data_dict['z_ref'] = z_ref_log
    data_dict['v_des'] = v_des_log
    data_dict['d'] = d_sf_log
    data_dict['solve_time'] = solve_time_log
    data_dict['error'] = error_log
    data_dict['primary_name'] = primary_controller.name
    data_dict['infeasible'] = infeasible
    data_dict['infeasible_index'] = infeasible_index
    if secondary_controllers is not None:
        data_dict['u_sec'] = u_secondary_log
        data_dict['v_sec'] = v_secondary_log
        data_dict['secondary_names'] = {}
        for ic, sec_controller in enumerate(secondary_controllers):
            data_dict['secondary_names'][ic] = sec_controller.name
    return data_dict, fig_count

