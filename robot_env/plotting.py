"""Matplotlib figures and notebook animations."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon
from matplotlib.animation import FuncAnimation


def plot_scene(env,rollout=None,waypoints=None,ax=None):
    if ax is None: _,ax=plt.subplots(figsize=(8,7))
    for o in env.obstacles:
        patch=Circle(o.center,o.radius) if o.radius is not None else Polygon(o.vertices)
        patch.set(facecolor='#64748b',edgecolor='#253449',alpha=.85);ax.add_patch(patch)
        ax.text(*o.center,o.name,ha='center',fontsize=9)
    if waypoints is not None:ax.plot(*np.asarray(waypoints).T,'o--',color='#d97706',label='Geometric route')
    if rollout is not None:
        ax.plot(*rollout.states[:,:2].T,color='#2563eb',label='Robot center')
        ax.plot(*rollout.outputs.T,color='#10b981',alpha=.6,label='Look-ahead output')
    p=env.state[:2];t=env.state[2]
    ax.add_patch(Circle(p,env.safety_radius,fill=False,color='#ef4444',linestyle='--'))
    ax.add_patch(Circle(p,env.robot.body_radius,color='#2563eb',alpha=.4))
    ax.arrow(*p,.15*np.cos(t),.15*np.sin(t),width=.008,color='#2563eb')
    xmin,xmax,ymin,ymax=env.bounds;ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel='x [m]',ylabel='y [m]',aspect='equal')
    ax.grid(alpha=.2)
    if rollout is not None or waypoints is not None:ax.legend(loc='upper left')
    return ax


def plot_signals(run):
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    axes[0].plot(run.times,run.states);axes[0].legend(['x [m]','y [m]','theta [rad]'])
    axes[1].step(run.times[:-1],run.wheels,where='post');axes[1].legend(['omega_R','omega_L']);axes[1].set_ylabel('rad/s')
    if run.margins.size:axes[2].plot(run.times,run.margins)
    axes[2].axhline(0,color='red',linestyle='--');axes[2].set(ylabel='Clearance margin [m]',xlabel='Time [s]')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();return fig


def animate(env,run,stride=4):
    if not isinstance(stride,int) or stride<1:raise ValueError('stride must be positive integer')
    fig,ax=plt.subplots(figsize=(7,6));plot_scene(env,run,ax=ax)
    disk=Circle(run.states[0,:2],env.robot.body_radius,color='#ef4444');ax.add_patch(disk)
    line,=ax.plot([],[],color='#ef4444'); title=ax.set_title('')
    def update(i):
        disk.center=run.states[i,:2];line.set_data(run.states[:i+1,0],run.states[:i+1,1]);title.set_text(f't = {run.times[i]:.2f} s')
        return disk,line,title
    frames=list(range(0,len(run.times),stride))
    if frames[-1]!=len(run.times)-1:frames.append(len(run.times)-1)
    return FuncAnimation(fig,update,frames=frames,interval=100,blit=False)
