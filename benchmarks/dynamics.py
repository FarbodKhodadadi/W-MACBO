"""Executed dynamics diagnostics for the physical differential-drive robot."""
import numpy as np
import pandas as pd
from .config import LABELS
from .metrics import scoring_intervals


def executed_dynamics_table(scenes, trials):
    """Time-weighted norms of xdot=f(x)+g(x)u, drift, and input matrix.

    The model has zero autonomous drift, not zero motion. Under each held
    wheel input, speed, yaw rate, and ||xdot|| are constant even as heading
    changes, so width-weighted norm averages are exact for this model.
    Successful terminal holds and censored failures follow benchmark scoring.
    No-command failures have undefined executed-motion means; model-field
    norms remain analytically known. Full-state L2 mixes position/angle units;
    separate speed and yaw columns provide physically interpretable units.
    """
    rows=[]
    for scenario, env in scenes.items():
        for name in LABELS:
            exp=trials[scenario][name].experiment
            states, wheels, widths, held=scoring_intervals(exp)
            if len(widths):
                derivatives=np.array([env.robot.f(q)+env.robot.g(q)@u
                                      for q,u in zip(states,wheels)])
                average=lambda values: float(np.average(values, weights=widths))
                motion=dict(mean_actual_dynamics_norm=average(np.linalg.norm(derivatives,axis=1)),
                            mean_planar_speed_m_s=average(np.linalg.norm(derivatives[:,:2],axis=1)),
                            mean_abs_yaw_rate_rad_s=average(np.abs(derivatives[:,2])))
                mean_f=average([np.linalg.norm(env.robot.f(q)) for q in states])
                mean_g=average([np.linalg.norm(env.robot.g(q),'fro') for q in states])
            else:
                motion=dict(mean_actual_dynamics_norm=np.nan,mean_planar_speed_m_s=np.nan,
                            mean_abs_yaw_rate_rad_s=np.nan)
                q=exp.rollout.states[0]
                mean_f=float(np.linalg.norm(env.robot.f(q)))
                mean_g=float(np.linalg.norm(env.robot.g(q),'fro'))
            rows.append(dict(scenario=scenario,method=name,label=LABELS[name],
                status=exp.rollout.status,scored_seconds=float(widths.sum()),
                terminal_hold=held,mean_f_norm=mean_f,mean_g_norm=mean_g,**motion))
    return pd.DataFrame(rows)


def dynamics_figure(frame, directory):
    """Two-panel IEEE-size figure: actual motion and model input matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from .plotting import COLORS, save_figure
    names={'three_circles':'Circles','u_and_t':'U + T',
           'central_star':'Star','square_passage':'Squares'}
    data=frame.assign(Scenario=frame.scenario.map(names))
    fig,axes=plt.subplots(1,2,figsize=(7.16,3.8))
    for ax,column,title,ylabel in zip(axes,
        ('mean_actual_dynamics_norm','mean_g_norm'),
        ('(a) Mean actual state-derivative norm','(b) Mean input-matrix norm'),
        ('Mean norm of f(x) + g(x)u [L2, unscaled]','Mean norm of g(x) [Frobenius, unscaled]')):
        sns.barplot(data=data,x='Scenario',y=column,hue='label',
            order=list(names.values()),hue_order=list(LABELS.values()),
            palette={LABELS[k]:COLORS[k] for k in LABELS},errorbar=None,ax=ax)
        ax.set(title=title,xlabel='Scenario',ylabel=ylabel)
        ax.grid(True,axis='both');ax.get_legend().remove()
    axes[0].text(.02,.98,'U + T: CLF–CBF mean undefined (no applied input)',
                 transform=axes[0].transAxes,va='top',fontsize=6.5)
    axes[0].set_ylim(0,float(frame.mean_actual_dynamics_norm.max())*1.24)
    axes[1].set_ylim(0,float(frame.mean_g_norm.max())*1.18)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=7)
    fig.suptitle('Actual robot dynamics across the four benchmark scenarios',fontsize=10)
    fig.subplots_adjust(bottom=.3,top=.83,wspace=.42)
    save_figure(fig,directory,'actual_robot_dynamics_norms')
    return fig
