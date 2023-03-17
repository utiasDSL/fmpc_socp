import os
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('darkgrid')

def get_data_dict(track_dir):
    data = np.load(os.path.join(track_dir, 'data.npz'), allow_pickle=True)
    data_dict = dict(zip(list(data.keys()), (d[()] for d in data.values())))
    return data_dict
def make_tracking_plot(track_dir, colors, save_name, labels=None, con=False, paper_dir=None, ctrls=None, error=False):
    data = get_data_dict(track_dir)

    fig_size = (5,3)

    # Plot the states along the trajectory and compare with reference.
    units = {0: 'm', 1: 'm/s', 2: 'm/s^2'}
    fig, ax = plt.subplots(figsize=fig_size)
    plt_id = 0
    alpha = 0.7
    if con:
        ax.axhline(y=con, color='k', linestyle='solid', label='Constraint')
    for ctrl_name, ctrl_data in data.items():
        if ctrls is None or ctrl_name in ctrls:
            if labels is None:
                label = ctrl_name
            else:
                label = labels[ctrl_name]
            common_plot_args = { 'label': label, 'color': colors[ctrl_name], 'alpha': alpha}
            if ctrl_data['infeasible']:
                inf_ind = ctrl_data['infeasible_index']
                if error:
                    ref = ctrl_data['z_ref'][:inf_ind, plt_id]
                    z_error = ctrl_data['z'][:inf_ind, plt_id] - ref
                    ax.plot(ctrl_data['t'][:inf_ind,:], z_error, **common_plot_args)
                    ax.plot(ctrl_data['t'][inf_ind-1,:], z_error[-1], 'rX', alpha=alpha)
                else:
                    ax.plot(ctrl_data['t'][:inf_ind,:], ctrl_data['z'][:inf_ind, plt_id], **common_plot_args)
                    ax.plot(ctrl_data['t'][inf_ind-1,:], ctrl_data['z'][inf_ind-1, plt_id], 'rX', alpha=alpha)
            else:
                if error:
                    z_error = ctrl_data['z_ref'][:, plt_id] - ctrl_data['z'][:, plt_id]
                    ax.plot(ctrl_data['t'], z_error, **common_plot_args)
                else:
                    ax.plot(ctrl_data['t'], ctrl_data['z'][:, plt_id], **common_plot_args)
    if not(error):
        ax.plot(ctrl_data['t'], ctrl_data['z_ref'][:, plt_id], '--', label='Reference', color=colors['ref'], zorder=1, alpha=alpha)
    y_label = f'$z_{plt_id}\; (' + units[plt_id] + ')$'
    if error:
        ax.set_ylabel('Position Error (m)')
    else:
        ax.set_ylabel('Horizontal Position (m)')
    ax.set_xlabel('Time (s)')
    ax.tick_params(labelsize=10)
    plt.legend()
    plt.tight_layout()
    plt_name = os.path.join(track_dir, save_name )
    plt.savefig(plt_name)
    if paper_dir is not None:
        plt_name = os.path.join(paper_dir, save_name )
        plt.savefig(plt_name)
    plt.show()

if __name__ == "__main__":

    colors = {'MPC': 'purple',
              'FMPC': 'lightsalmon',
              'DLQR':  'skyblue',
              'FMPC+SOCP': 'orangered',
              'DLQR+SOCP': 'royalblue',
              'GPMPC': 'darkgreen',
              'ref': 'dimgray'
              }
    labels = {'MPC': 'MPC',
              'FMPC': 'FMPC',
              'DLQR':  'DLQR',
              'FMPC+SOCP': 'FMPC+SOCP (ours)',
              'DLQR+SOCP': 'DLQR+SOCP',
              'GPMPC': 'GPMPC',
              'ref': 'Reference'
              }
    paper_fig_dir = '/home/ahall/Documents/UofT/papers/input_and_stat_constrained_SOCP_filter_overleaf/figs'
    tracking_dir = '/home/ahall/Documents/UofT/code/dsl__projects__flatness_safety_filter/testing/results/tracking_comp/saved/seed42_Mar-10-01-49-38_fe35b85'
    tracking_name = 'tracking.pdf'
    ctrls = ['MPC', 'DLQR', 'GPMPC', 'FMPC+SOCP']
    make_tracking_plot(tracking_dir, colors, tracking_name, labels=labels, paper_dir=paper_fig_dir, ctrls=ctrls, error=True)
    step_dir = '/home/ahall/Documents/UofT/code/dsl__projects__flatness_safety_filter/testing/results/tracking_comp/saved/seed42_Mar-09-23-47-14_fe35b85'
    step_name = 'step.pdf'
    make_tracking_plot(step_dir, colors, step_name, labels=labels, paper_dir=paper_fig_dir)
    con_step_dir = '/home/ahall/Documents/UofT/code/dsl__projects__flatness_safety_filter/testing/results/tracking_comp/saved/seed42_Mar-11-13-24-20_7c1ada4'
    con_step_name = 'constrained_step.pdf'
    make_tracking_plot(con_step_dir, colors, con_step_name, labels=labels, con=0.51, paper_dir=paper_fig_dir)
    input_con_step_dir = '/home/ahall/Documents/UofT/code/dsl__projects__flatness_safety_filter/testing/results/tracking_comp/saved/seed42_Mar-17-01-52-47_7c1ada4'
    input_con_step_name = 'input_constrained_step.pdf'
    colors['DLQR'] = 'brown'
    labels['DLQR'] = 'DLQR Known'
    make_tracking_plot(input_con_step_dir, colors, input_con_step_name, labels=labels, con=None, paper_dir=paper_fig_dir)
