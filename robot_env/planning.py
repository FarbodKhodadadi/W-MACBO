"""Baseline geometric A* and pose-aware waypoint follower, not MACBO."""
import heapq
import itertools
import numpy as np
from .core import vector, positive


def astar(env,start,goal,spacing=0.15,extra=0.03):
    """Four-neighbor Euclidean-cost A*, exact disk/segment collision tests.

    Bounds constrain this planner only. Endpoint links connect to all visible
    nodes within 2*spacing. Raises on failure; caller can refine the grid.
    This baseline omits the paper's turn-augmented cost and long grid edges.
    """
    start,goal=vector(start,2),vector(goal,2); positive(spacing)
    if not env.is_free(start) or not env.is_free(goal): raise ValueError('Unsafe endpoint')
    if env.segment_free(start,goal,extra): return np.array([start,goal])
    xmin,xmax,ymin,ymax=env.bounds
    for p in [start,goal]:
        if not (xmin<=p[0]<=xmax and ymin<=p[1]<=ymax): raise ValueError('Endpoint outside planning bounds')
    xs=np.arange(xmin,xmax+1e-12,spacing);ys=np.arange(ymin,ymax+1e-12,spacing)
    nodes={(i,j):np.array([x,y]) for i,x in enumerate(xs) for j,y in enumerate(ys) if np.all(env.margins((x,y))>=extra)}
    endpoints={}
    for key,p in [('start',start),('goal',goal)]:
        endpoints[key]=[n for n,q in nodes.items() if np.linalg.norm(p-q)<=2*spacing and env.segment_free(p,q,extra)]
        if not endpoints[key]: raise RuntimeError('No endpoint attachment; reduce spacing or extra')
    nodes['start']=start;nodes['goal']=goal; goal_links=set(endpoints['goal'])
    def neighbors(n):
        if n=='start': return endpoints['start']
        if n=='goal': return []
        i,j=n; result=[m for m in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)] if m in nodes]
        if n in goal_links: result.append('goal')
        return result
    seq=itertools.count(); queue=[(0,next(seq),'start')];cost={'start':0.};parent={};closed=set()
    while queue:
        _,_,n=heapq.heappop(queue)
        if n in closed: continue
        if n=='goal':
            route=[goal]; cur=n
            while cur!='start':cur=parent[cur];route.append(nodes[cur])
            route=route[::-1]
            # Greedy visible shortening; preserves clearance, not grid optimality.
            result=[route[0]];i=0
            while i<len(route)-1:
                j=len(route)-1
                while j>i+1 and not env.segment_free(route[i],route[j],extra):j-=1
                result.append(route[j]);i=j
            return np.array(result)
        closed.add(n)
        for m in neighbors(n):
            if not env.segment_free(nodes[n],nodes[m],extra):continue
            c=cost[n]+np.linalg.norm(nodes[m]-nodes[n])
            if c<cost.get(m,float('inf')):
                cost[m]=c;parent[m]=n
                heapq.heappush(queue,(c+np.linalg.norm(nodes[m]-goal),next(seq),m))
    raise RuntimeError('No grid route; refine spacing or change map')


class WaypointFollower:
    """Simple stop/turn/drive baseline. Final heading is controlled explicitly."""
    def __init__(self,waypoints,goal_heading=0.,speed=0.12,tolerance=0.035):
        self.waypoints=np.asarray(waypoints,dtype=float)
        if self.waypoints.ndim!=2 or self.waypoints.shape[1]!=2 or len(self.waypoints)<2 or not np.all(np.isfinite(self.waypoints)):raise ValueError('Need at least two finite waypoints')
        self.heading=float(goal_heading);self.speed=positive(speed);self.tolerance=positive(tolerance);self.index=1

    def __call__(self,obs,env):
        p=obs['position'];theta=obs['state'][2]
        while self.index<len(self.waypoints)-1 and np.linalg.norm(p-self.waypoints[self.index])<self.tolerance:self.index+=1
        delta=self.waypoints[self.index]-p; dist=np.linalg.norm(delta)
        target=self.heading if self.index==len(self.waypoints)-1 and dist<self.tolerance else np.arctan2(delta[1],delta[0])
        error=np.arctan2(np.sin(target-theta),np.cos(target-theta))
        w=np.clip(3*error,-1.5,1.5)
        v=min(self.speed,dist) if abs(error)<0.15 and dist>=self.tolerance else 0.
        return np.array([v,w])
