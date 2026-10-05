"""Vectorized geometry and smooth convex-component barriers, without core edits."""
import numpy as np


def signed_distances(points, obstacles):
    """Return (N,M) signed distances matching robot_env.Obstacle."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    columns = []
    for o in obstacles:
        if o.radius is not None:
            columns.append(np.linalg.norm(p-o.center, axis=1)-o.radius)
            continue
        v = o.vertices
        distance = np.full(len(p), np.inf)
        inside = np.zeros(len(p), dtype=bool)
        for a, b in zip(v, np.roll(v, -1, axis=0)):
            d = b-a
            t = np.clip((p-a)@d/np.dot(d, d), 0, 1)
            distance = np.minimum(distance, np.linalg.norm(p-a-t[:, None]*d, axis=1))
            if b[1] != a[1]:
                crosses = (a[1] > p[:, 1]) != (b[1] > p[:, 1])
                xcross = (b[0]-a[0])*(p[:, 1]-a[1])/(b[1]-a[1])+a[0]
                inside ^= crosses & (p[:, 0] < xcross)
        columns.append(np.where(inside, -distance, distance))
    return np.column_stack(columns) if columns else np.empty((len(p), 0))


def triangulate(vertices):
    """Ear clipping of a simple polygon; retains concave cavities exactly."""
    v = np.asarray(vertices, dtype=float)
    if len(v) < 3 or not np.all(np.isfinite(v)):
        raise ValueError('Invalid polygon')
    cross = lambda a,b: a[0]*b[1]-a[1]*b[0]
    area = sum(cross(a,b) for a,b in zip(v,np.roll(v,-1,axis=0)))
    if area < 0: v = v[::-1]
    indices = list(range(len(v)))
    triangles = []
    while len(indices) > 3:
        for k in range(len(indices)):
            ia,ib,ic = indices[k-1],indices[k],indices[(k+1)%len(indices)]
            a,b,c = v[[ia,ib,ic]]
            if cross(b-a,c-b) <= 1e-13: continue
            def contained(p):
                return min(cross(b-a,p-a),cross(c-b,p-b),cross(a-c,p-c)) >= -1e-13
            if any(contained(v[j]) for j in indices if j not in (ia,ib,ic)): continue
            triangles.append(np.array([a,b,c])); indices.pop(k); break
        else:
            raise ValueError('Polygon is degenerate or not simple; decomposition failed')
    triangles.append(v[indices])
    return triangles


class Barriers:
    """C1 squared-distance barriers for circles and convex polygon components.

    Circle b=||p-c||²-(R+r)²; triangle b=dist(p,C)²-r².
    Intersection of b>=0 equals clearance to the original occupied union.
    """
    def __init__(self, env):
        self.radius = env.safety_radius
        self.components = []
        for o in env.obstacles:
            if o.radius is not None: self.components.append(('circle',o.center,o.radius))
            else:
                for triangle in triangulate(o.vertices): self.components.append(('triangle',triangle,None))

    def values(self, points):
        p = np.asarray(points,dtype=float).reshape(-1,2)
        columns=[]
        for kind,data,radius in self.components:
            if kind=='circle':
                columns.append(np.sum((p-data)**2,axis=1)-(radius+self.radius)**2)
                continue
            distance2=np.full(len(p),np.inf); inside=np.ones(len(p),dtype=bool)
            for a,b in zip(data,np.roll(data,-1,axis=0)):
                d=b-a; offset=p-a
                inside &= d[0]*offset[:,1]-d[1]*offset[:,0] >= -1e-14
                t=np.clip(offset@d/np.dot(d,d),0,1)
                distance2=np.minimum(distance2,np.sum((offset-t[:,None]*d)**2,axis=1))
            columns.append(np.where(inside,0.,distance2)-self.radius**2)
        return np.column_stack(columns) if columns else np.empty((len(p),0))


def softmin(values,eta):
    """Unnormalized stable -log(sum(exp(-eta*b)))/eta, never a soft average."""
    from scipy.special import logsumexp
    values=np.asarray(values,dtype=float)
    if values.shape[-1]==0: return np.full(values.shape[:-1],np.inf)
    return -logsumexp(-eta*values,axis=-1)/eta


def dhocbf_levels(values,gammas):
    """ψ[j+1,k]=ψ[j,k+1]-(1-γ[j+1])ψ[j,k], all lower levels retained."""
    current=np.asarray(values,dtype=float)
    levels=[current]
    for gamma in gammas:
        current=current[1:]-(1-gamma)*current[:-1]
        levels.append(current)
    return levels
