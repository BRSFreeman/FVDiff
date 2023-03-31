import numpy as np
from numpy.linalg import norm, det, inv, LinAlgError, lstsq
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from scipy.interpolate import LinearNDInterpolator

from numba import jit, njit

from tqdm import trange


class greenGaussBase:
    def __init__(self, mesh, dim='2d'):
        self.points = mesh.points
        self.n_points = len(mesh.points)
        self.vertices = mesh.vertices
        self.n_vertices = len(mesh.vertices)
        self.regions = mesh.regions
        self.n_regions = len(mesh.regions)
        self.ridge_points = mesh.ridge_points
        self.n_ridges = len(mesh.ridge_points)
        self.ridge_vertices = mesh.ridge_vertices
        self.point_region = mesh.point_region

        self.neighbours = [[] for i in range(self.n_points)]
        self.n_neighbours = np.zeros(self.n_points)

        for points in self.ridge_points:
            self.neighbours[points[0]].append(points[1])
            self.neighbours[points[1]].append(points[0])
            self.n_neighbours[points[0]] += 1
            self.n_neighbours[points[1]] += 1

        if dim == '2d':
            self.ndim = 2
            self.area_func = area2d
            self.interp = _partial_interp

        elif dim == '3d':
            self.ndim = 3
            self.area_func = area3d
            self.interp = _partial_interp3d
        else:
            raise ValueError(f'number of dimensions {dim} not recognized. Options are "2d" or "3d"')

        self.volume()


    def volume(self):
        self.vol = np.zeros(self.n_regions)

        if self.ndim == 2:
            for i in range(self.n_regions):
                region = self.regions[i]
                if -1 in region:
                    continue
                elif len(region)==0:
                    self.vol[i] = 0
                else:
                    self.vol[i] = div_area(self.vertices[region])

        elif self.ndim == 3:
            # https://en.wikipedia.org/wiki/Polyhedron#Volume
            for i in range(self.n_ridges):
                vertices = self.ridge_vertices[i]

                point_idx = self.ridge_points[i]
                pt = self.vertices[vertices[0]]

                area = area3d(self, vertices)
                d_points = self.points[point_idx[1]] - self.points[point_idx[0]]
                normal = d_points / norm(d_points)

                contr = ((pt@normal)*area / 3)

                self.vol[self.point_region[point_idx[0]]] += contr
                self.vol[self.point_region[point_idx[1]]] -= contr

    def green_gauss(self, q, out_shape):
        # iterate over faces to calculate flux in or out
        out = np.zeros(out_shape)

        for ridge in range(self.n_ridges):
            print(ridge)
            vertices = self.ridge_vertices[ridge]

            # exclude infinite vertices
            if -1 in vertices:
                continue

            point_idx = self.ridge_points[ridge]

            area = self.area_func(self, vertices)

            d_points = self.points[point_idx[1]] - self.points[point_idx[0]]
            normal = d_points / norm(d_points)

            face_centr = mean(self.vertices[vertices], axis=0)  # centroid of current face

            shared_neighbours = _establish_neighbours(self.neighbours, point_idx)

            face_q = self.interp(
                self.points,
                self.vertices,
                q,
                point_idx,
                np.array(vertices),
                np.array(shared_neighbours),
                face_centr
            )

            face_flux = self.operator(face_q, normal, area)

            out[self.point_region[point_idx[0]]] += face_flux
            out[self.point_region[point_idx[1]]] -= face_flux

        out = (out.T / self.vol).T

        return out


class gaussDiv(greenGaussBase):
    def __init__(self, mesh, dim='2d'):
        super().__init__(mesh, dim)

    def operator(self, face_q, normal, area):
        return (face_q@normal) * area
    
    def div(self, q):
        return self.green_gauss(q, self.n_regions)

    def __call__(self, q):
        return self.div(q)

    
class gaussGrad(greenGaussBase):
    def __init__(self, mesh, dim='2d'):
        super().__init__(mesh, dim)
        
    def operator(self, face_q, normal, area):
        return face_q*normal*area
    
    def grad(self, q):
        return self.green_gauss(q, (self.n_regions, self.ndim))

    def __call__(self, q):
        return self.grad(q)

    
class gaussLaplacian(greenGaussBase):
    def __init__(self, mesh, dim='2d'):
        super().__init__(mesh, dim)
        self.interp = _lstsq_grad
        
    def operator(self, face_q, normal, area):
        return (face_q@normal) * area
    
    def laplace(self, q):
        return self.green_gauss(q, self.n_regions)

    def __call__(self, q):
        return self.laplace(q)

    
def _establish_neighbours(neighbours, pts):
    shared_neighbours = list(
        (set(neighbours[pts[0]])
        & set(neighbours[pts[1]]))
    )
    if len(shared_neighbours) == 0:
        shared_neighbours = list(
            (set(neighbours[pts[0]]) - {pts[1]})
            | (set(neighbours[pts[1]]) - {pts[0]})
        )
    return shared_neighbours


@njit
def _lstsq_grad(points, vertices, u, pts, ridge, neighbours, pt):
    """calculate least squares gradient based on nearest neighbours
       e.g. GG-LSQ https://doi.org/10.1186/s42774-019-0020-9"""
    cells = np.array(list(set(pts) | set(neighbours)))

    x_n = points[cells]

    u_n = u[cells]

    dx = x_n - pt

    Rinv = 1/np.sqrt(np.sum(dx**2, axis=1))
    
    A = np.zeros((dx.shape[0], dx.shape[1]+1))
    A[:, 0] = Rinv
    A[:, 1:] = (Rinv*dx.T).T
    b = (Rinv*u_n.T).T

    x, _, _, _ = lstsq(A, b)

    return x[1:]


def _partial_interp(points, vertices, u, pts, ridge, neighbours, pt):
    """interpolate based on the Delaunay simplices shared by 
       two points of a Voronoi diagram
       
       Close to GG-WTLI of https://doi.org/10.1186/s42774-019-0020-9
    """
    
    x0 = points[pts[0]]
    x1 = points[pts[1]]

    # find the simplex the interpolation point is in
    for i in neighbours:
        xi = points[i]
        
        lamb = _barycentric_coords(
            np.stack([x0, x1, xi], axis=1),
            pt
        )
        
        if np.all(lamb >= 0):
            return lamb[0]*u[pts[0]] + lamb[1]*u[pts[1]] + lamb[2]*u[i]
    
    # return interpolation based on last simplex if the point is outside
    # return lamb[0]*u[p0] + lamb[1]*u[p1] + lamb[2]*u[i]
    return 1/2*(u[pts[0]] + u[pts[1]])


@njit
def _partial_interp3d(points, vertices, u, pts, ridge, neighbours, pt):
    """Find and interpolate based on a simplex
    
       Close to GG-WTLI of https://doi.org/10.1186/s42774-019-0020-9
    """
    # Find simplex containing a particular point on a ridge in Voronoi diagram
    x_0 = points[pts[0]]
    x_n = points[neighbours]

    for vertex in ridge:
        # find simplex associated with vertex
        v = vertices[vertex]

        dist_to_neighbours = np.sum((x_n - v)**2, axis=1)
        closest_neighbours = np.argsort(dist_to_neighbours)

        simplex_points = x_n[closest_neighbours[:2]]

        # check if centroid is in simplex
        l = _barycentric_coords(
            np.vstack((points[pts], simplex_points)).T,
            pt
        )

        if np.all(l >= 0):  # centroid is in simplex - interpolate

            u_simp = np.vstack((u[pts], u[closest_neighbours[:2]]))
            return (
                l[0]*u[pts[0]]
                + l[1]*u[pts[1]]
                + l[2]*u[closest_neighbours[0]]
                + l[3]*u[closest_neighbours[1]]
            )

    # if centroid isn't in any simplex, approximate as mean of points
    # this typically happens outside the domain
    return 1/2*(u[pts[0]] + u[pts[1]])


@njit
def _barycentric_coords(pts, xi, tol=1e-9):
    
    ndim = pts.shape[1]
    A = np.vstack((np.ones((1, ndim)), pts))
    
    b = np.concatenate((np.ones(1), xi))

    if abs(det(A)) <= tol:
        return np.array([-1., 0, 0])

    return inv(A)@b


def area2d(mesh, vertex_idx):
    
    p1 = mesh.vertices[vertex_idx[0]]
    p2 = mesh.vertices[vertex_idx[1]]

    return norm(p2 - p1)
        

def area3d(mesh, vertex_idx):
    
    points = mesh.vertices[vertex_idx]
    
    return _3d_area(points)

@njit
def _3d_area(points):
    area = 0
    n_points = points.shape[0]
    
    for i in range(1, n_points-1):
        d12 = points[i] - points[0]
        d23 = points[i+1] - points[i]

        area += 1/2*norm(cross3(d12, d23))
    
    return area
    
@njit
def cross3(a, b):
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

@njit
def div_area(x):
    
    dxy = x[0] - x[-1]

    a = x[0, 0]*dxy[1] - x[0, 1]*dxy[0]

    for i in range(len(x)-1):
        dxy = x[i+1] - x[i]

        a = a + x[i, 0]*dxy[1] - x[i, 1]*dxy[0]
    
    return abs(a) / 2
    

@njit
def norm(x):
    return np.sqrt(np.sum(x**2))


@njit
def mean(x, axis=0):
    return np.sum(x, axis) / x.shape[axis]

