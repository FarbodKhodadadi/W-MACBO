"""Seaborn/Times New Roman figures and Pillow GIFs, with explicit failure markers."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
from .config import LABELS, HORIZON
from .scenarios import START, GOAL, TITLES

COLORS=dict(zip(LABELS,sns.color_palette('colorblind',7)))
STYLES=['-','--','-.',':',(0,(5,1,1,1)),(0,(3,1)),'-']


def publication_style():
    """Require the requested font rather than silently substituting it."""
    font_manager.findfont('Times New Roman',fallback_to_default=False)
    sns.set_theme(style='whitegrid',context='paper',font='Times New Roman')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,
        'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,
        'xtick.labelsize':8,'ytick.labelsize':8,'lines.linewidth':1.2,
        'axes.linewidth':.6,'grid.linewidth':.4,'grid.alpha':.45,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':600,
        'figure.dpi':110,'mathtext.fontset':'custom',
        'mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic',
        'mathtext.bf':'Times New Roman:bold'})


def draw_map(ax,environment):
    for obstacle in environment.obstacles:
        patch=Circle(obstacle.center,obstacle.radius) if obstacle.radius is not None else Polygon(obstacle.vertices)
        patch.set(facecolor='#bfc5cc',edgecolor='#4b5563',alpha=.85,linewidth=.6)
        ax.add_patch(patch)
    ax.plot(*START[:2],marker='o',color='black',markersize=3,label='_nolegend_')
    ax.plot(*GOAL[:2],marker='*',color='black',markersize=7,label='_nolegend_')
    ax.set(xlim=environment.bounds[:2],ylim=environment.bounds[2:],aspect='equal',
           xlabel='x [m]',ylabel='y [m]')
    ax.grid(True)


def save_figure(fig,directory,stem):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/f'{stem}.png',dpi=600,bbox_inches='tight',facecolor='white')
    fig.savefig(directory/f'{stem}.pdf',bbox_inches='tight',facecolor='white')


def legend_below(ax):
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3,frameon=False)


def comparison_figures(scenario,environment,trials,directory):
    """Return three separate, double-column-sized figures per map."""
    figures=[]
    fig,ax=plt.subplots(figsize=(7.16,5.3));draw_map(ax,environment)
    for index,name in enumerate(LABELS):
        trial=trials[name];run=trial.experiment.rollout
        sns.lineplot(x=run.states[:,0],y=run.states[:,1],ax=ax,estimator=None,
                     sort=False,color=COLORS[name],linestyle=STYLES[index],label=LABELS[name])
        if run.status!='goal':
            ax.plot(*run.states[-1,:2],marker='x',color=COLORS[name],markersize=5)
        if name=='macbo' and trial.graph_path is not None:
            ax.scatter(*trial.graph_path.T,s=2,color=COLORS[name],alpha=.65,zorder=4,
                       label='_nolegend_')
            ax.scatter(*trial.waypoints.T,s=9,facecolors='none',edgecolors=COLORS[name],
                       linewidths=.5,zorder=4,label='_nolegend_')
    ax.set_title(TITLES[scenario]+' — trajectories')
    legend_below(ax);fig.subplots_adjust(bottom=.22)
    save_figure(fig,directory,'trajectory');figures.append(fig)
    for quantity in ('control_norm','orientation'):
        fig,ax=plt.subplots(figsize=(7.16,3.8))
        for index,name in enumerate(LABELS):
            run=trials[name].experiment.rollout
            if quantity=='control_norm':
                if len(run.wheels):
                    values=np.linalg.norm(run.wheels,axis=1)
                    times=np.r_[run.times[:-1],run.times[-1]]
                    values=np.r_[values,0. if run.status=='goal' else values[-1]]
                    if run.status=='goal' and times[-1]<HORIZON:
                        times=np.r_[times,HORIZON];values=np.r_[values,0.]
                    ax.step(times,values,where='post',color=COLORS[name],linestyle=STYLES[index],label=LABELS[name])
                else:ax.plot([],[],color=COLORS[name],linestyle=STYLES[index],label=LABELS[name]+' (no input)')
            else:
                times=run.times.copy();values=np.arctan2(np.sin(run.states[:,2]),np.cos(run.states[:,2]))
                if run.status=='goal' and times[-1]<HORIZON:
                    times=np.r_[times,HORIZON];values=np.r_[values,values[-1]]
                # Break the plotted line at wrap discontinuities; values are not smoothed.
                indices=np.flatnonzero(np.abs(np.diff(values))>np.pi)+1
                times=np.insert(times,indices,np.nan);values=np.insert(values,indices,np.nan)
                ax.plot(times,values,color=COLORS[name],linestyle=STYLES[index],label=LABELS[name])
            if run.status!='goal' and len(run.wheels):
                value=np.linalg.norm(run.wheels[-1]) if quantity=='control_norm' else np.arctan2(np.sin(run.states[-1,2]),np.cos(run.states[-1,2]))
                ax.plot(run.times[-1],value,'x',color=COLORS[name],markersize=4)
        title='wheel-control norm' if quantity=='control_norm' else 'wrapped orientation'
        ax.set(title=TITLES[scenario]+' — '+title,xlabel='Time [s]',xlim=(0,HORIZON),
               ylabel='Wheel-control norm [rad/s]' if quantity=='control_norm' else 'Orientation θ [rad]')
        if quantity=='orientation':
            ax.axhline(GOAL[2],color='black',linewidth=.7,linestyle=':',label='_nolegend_')
            ax.set_ylim(-np.pi-.15,np.pi+.15)
        ax.grid(True);legend_below(ax);fig.subplots_adjust(bottom=.28)
        save_figure(fig,directory,quantity);figures.append(fig)
    return figures


def pose_at(environment,run,time):
    if time>=run.times[-1] or not len(run.wheels):return run.states[-1].copy()
    index=max(0,int(np.searchsorted(run.times,time,side='right')-1))
    if time<=run.times[index]+1e-12:return run.states[index].copy()
    return environment.robot.advance(run.states[index],run.wheels[index],time-run.times[index])


def motion_frame_times(environment, run, *, max_time_step=.4, max_heading_step=.25):
    """Bound physical-time gaps and accumulated yaw between animation frames.

    Exact physical poses are reconstructed at these times. Playback spends
    extra frames on fast turns; the displayed simulation clock is authoritative.
    """
    if max_time_step<=0 or max_heading_step<=0:
        raise ValueError('Frame resolutions must be positive')
    times=[float(run.times[0])]
    angle_budget=max_heading_step
    for start,end,u in zip(run.times[:-1],run.times[1:],run.wheels):
        yaw=abs(environment.robot.wheels_to_twist(u)[1])
        cursor=float(start)
        while cursor<end-1e-12:
            until_time=max_time_step-(cursor-times[-1])
            until_angle=angle_budget/yaw if yaw>1e-14 else np.inf
            width=min(float(end-cursor),until_time,until_angle)
            cursor+=width
            angle_budget-=yaw*width
            if until_time<=width+1e-12 or until_angle<=width+1e-12:
                times.append(cursor)
                angle_budget=max_heading_step
    if run.times[-1]>times[-1]+1e-12:times.append(float(run.times[-1]))
    return np.array(times)



def save_motion_gif(environment,trial,name,path,fps=16,frames=None):
    """Exact accepted motion, resolved turns, visible failures, endpoint pause.

    `frames` is a legacy minimum frame count, not a cap on turn resolution.
    A JSON sidecar saves physical sample times/poses and outcome. Terminal
    frames are held for 1.5 playback seconds; identical GIF frames may merge.
    """
    import json, textwrap
    run=trial.experiment.rollout
    fig,ax=plt.subplots(figsize=(6,6));draw_map(ax,environment)
    color=COLORS[name]
    if name=='macbo' and trial.graph_path is not None:
        ax.scatter(*trial.graph_path.T,s=2,color=color,label='Initial A* points')
    line,=ax.plot([],[],color=color,label=LABELS[name],linewidth=1.4)
    disk=Circle(run.states[0,:2],environment.robot.body_radius,color=color)
    safety=Circle(run.states[0,:2],environment.safety_radius,fill=False,color=color,linestyle=':',linewidth=.7)
    ax.add_patch(disk);ax.add_patch(safety)
    heading,=ax.plot([],[],color='black',linewidth=1.)
    ax.set_title('U + T — '+LABELS[name]);ax.legend(loc='upper left',fontsize=8)
    text=ax.text(.02,.02,'',transform=ax.transAxes,fontsize=8,
                 bbox=dict(facecolor='white',alpha=.95,edgecolor='none'))
    movie_times=motion_frame_times(environment,run)
    if frames and run.times[-1]>0:
        movie_times=np.unique(np.r_[movie_times,np.linspace(run.times[0],run.times[-1],frames)])
    # A distinct opening card followed by the held final outcome also makes
    # zero-command failures a valid multi-frame GIF without invented motion.
    timeline=[(float(movie_times[0]),True)]
    timeline.extend((float(t),False) for t in movie_times)
    timeline.extend([(float(movie_times[-1]),False)]*int(round(1.5*fps)))
    reason=trial.experiment.failure or ('Horizon exhausted' if run.status!='goal' else '')
    def update(frame):
        time,opening=frame
        state=pose_at(environment,run,time)
        disk.center=state[:2];safety.center=state[:2]
        index=int(np.searchsorted(run.times,time,side='right'))
        trace=np.vstack((run.states[:index,:2],state[:2]))
        line.set_data(trace[:,0],trace[:,1])
        tip=state[:2]+.18*np.array([np.cos(state[2]),np.sin(state[2])])
        heading.set_data([state[0],tip[0]],[state[1],tip[1]])
        finished=time>=run.times[-1]-1e-10
        status='recorded start' if opening else (run.status if finished else 'running')
        detail='Adaptive playback; physical time shown above'
        if finished and not opening:
            detail='Goal reached' if run.status=='goal' else textwrap.fill(str(reason),width=57)
            if not len(run.wheels):detail='No command applied. '+detail
        text.set_text(f't = {time:.2f} s | {status}\n'+detail)
        return line,disk,safety,heading,text
    movie=FuncAnimation(fig,update,frames=timeline,interval=1000/fps,blit=False,
                        cache_frame_data=False)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    movie.save(path,writer=PillowWriter(fps=fps),dpi=90)
    plt.close(fig)
    path.with_suffix('.json').write_text(json.dumps(dict(
        status=run.status,failure=reason,fps=fps,endpoint_pause_seconds=1.5,
        playback='adaptive; sample intervals vary to resolve turns',
        physical_times=movie_times.tolist(),
        physical_poses=[pose_at(environment,run,t).tolist() for t in movie_times]),indent=2))
    return path


def table_style(frame,caption):
    return (frame.style.format(precision=3,na_rep='—').set_caption(caption)
        .set_table_styles([
            {'selector':'','props':[('font-family','Times New Roman'),('font-size','12px'),('border-collapse','collapse')]},
            {'selector':'caption','props':[('font-family','Times New Roman'),('font-size','15px'),('text-align','left'),('padding','8px 0')]},
            {'selector':'th','props':[('border-bottom','1px solid #444'),('text-align','right'),('padding','6px')]},
            {'selector':'td','props':[('padding','5px 7px'),('text-align','right')]},
            {'selector':'tbody tr:nth-child(even)','props':[('background-color','#f3f5f7')]}]))
