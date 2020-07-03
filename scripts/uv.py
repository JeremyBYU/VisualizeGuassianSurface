import math
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from .normals import create_samples, create_arrows


def flatten(l): return [item for sublist in l for item in sublist]


class VertexHolder(object):
    def __init__(self):
        self.vertex_dict = dict()
        self.all_vertices = []

    @staticmethod
    def get_key(point):
        "Returns a string key for a vertex"
        return f"{point[0]:.4f}:{point[1]:4f}:{point[2]:4f}"

    def add_vertex(self, point):
        key = self.get_key(point)
        if key in self.vertex_dict:
            return self.vertex_dict[key]
        else:
            # Need to add this vertex
            self.all_vertices.append(point)
            self.vertex_dict[key] = len(self.all_vertices) - 1
            return len(self.all_vertices) - 1


def set_axes_radius(ax, origin, radius, ignore_z=False):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    if not ignore_z:
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def polar_to_xyz(phi, theta):
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)
    return [x, y, z]


def spread_discretization(phi_num=20):
    phis = np.linspace(0, math.pi, phi_num + 1, endpoint=True)
    vh = VertexHolder()
    faces_all = []
    cells_all = []
    for i in range(1, phi_num):
        phi = phis[i]
        nc = int(np.round(2 * phi_num * math.sin(phi)))
        thetas = np.linspace(0, 2 * np.pi, nc, endpoint=False)

        phi_top = (phis[i - 1] + phis[i]) / 2.0
        phi_bottom = (phis[i + 1] + phis[i]) / 2.0
        theta_step = (thetas[1] - thetas[0]) / 2.0
        faces = []
        cells = []
        for theta in thetas:
            theta_left = theta - theta_step
            theta_right = theta + theta_step

            p_center = polar_to_xyz(phi, theta)
            cells.append(p_center)

            p1 = polar_to_xyz(phi_top, theta_left)
            p2 = polar_to_xyz(phi_top, theta_right)
            p3 = polar_to_xyz(phi_bottom, theta_right)
            p4 = polar_to_xyz(phi_bottom, theta_left)

            # so that there is a curve between p3 to p4
            bunch_of_thetas = np.linspace(theta_right, theta_left, num=10, endpoint=False)
            bunch_of_points = [polar_to_xyz(phi_bottom, b_t) for b_t in bunch_of_thetas[1:]]

            p_indices = [vh.add_vertex(p) for p in [p1, p2, p3, *bunch_of_points, p4]]
            faces.append(p_indices)
        cells_all.append(cells)
        faces_all.append(np.array(faces))

    return np.row_stack(vh.all_vertices), faces_all, cells_all


def set_axes_equal(ax, ignore_z=False):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius, ignore_z=ignore_z)


def uv_discretization(phi=20, theta=20, start_phi=0.01):
    phis = [i * (math.pi) / phi + start_phi for i in range(phi)]
    thetas = [i * (2 * math.pi) / theta for i in range(theta)]

    return phis, thetas


def create_xyz_grid(thetas, phis):
    shape = (len(thetas), len(phis))
    grid = np.zeros(shape + (3, ))
    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            # import ipdb; ipdb.set_trace()
            x = math.sin(phi) * math.cos(theta)
            y = math.sin(phi) * math.sin(theta)
            z = math.cos(phi)
            grid[i, j] = [x, y, z]

    return grid, shape


def create_cells(grid):
    vh = VertexHolder()
    faces = []
    max_row = grid.shape[0]
    max_column = grid.shape[1]
    for i in range(max_row):
        for j in range(max_column):
            if i < max_row - 1 and j < max_column - 1:
                p1 = grid[i, j, :]
                p2 = grid[i + 1, j, :]
                p3 = grid[i + 1, j + 1, :]
                p4 = grid[i, j + 1, :]
                p_indices = [vh.add_vertex(p) for p in [p1, p2, p3, p4]]
                faces.append(p_indices)
            else:
                if i == max_row - 1 and j < max_column - 1:
                    p1 = grid[i, j, :]
                    p2 = grid[0, j, :]
                    p3 = grid[0, j + 1, :]
                    p4 = grid[i, j + 1, :]
                    p_indices = [vh.add_vertex(p) for p in [p1, p2, p3, p4]]
                    faces.append(p_indices)
                else:
                    pass
                    # something should be here....

    return np.row_stack(vh.all_vertices), np.array(faces)


def draw_faces(vertices: np.ndarray, faces: np.ndarray, ax: mpl.axes.Axes):
    faces_points = vertices[faces]
    colors = plt.get_cmap('tab20').colors
    all_polys = []
    for i in range(faces_points.shape[0]):
        xyz = faces_points[i, :, :]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='k', alpha=0.0)
        all_polys.append(xyz.tolist())

    p3d = Poly3DCollection(all_polys)
    p3d.set_color(colors[1])
    p3d.set_edgecolor('k')
    p3d.set_alpha(1.0)
    ax.add_collection3d(p3d)


def create_samples_and_arrow(ax, normal=np.array([0, 1, 0]), size=50, **kwargs):
    normals_3d = create_samples(normal=normal, size=size, **kwargs)
    create_arrows(ax, normals_3d)
    return normals_3d


def plot(vertices, faces_list):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')

    for faces in faces_list:
        draw_faces(vertices, faces, ax)
    set_axes_equal(ax)

    samples_z = create_samples_and_arrow(ax, normal=np.array([0, 0, 1]), size=100)
    samples_y = create_samples_and_arrow(ax, normal=np.array([0, 1, 0]), size=100)

    # Hide axes ticks
    plt.axis('off')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # get correct view
    ax.view_init(19, -73)
    ax.dist = 6.4
    plt.show()


def main():
    lats, longs = uv_discretization(phi=20, theta=20)
    grid, shape = create_xyz_grid(longs, lats)
    vertices, faces = create_cells(grid)

    plot(vertices, [faces])

    vertices, faces, cells = spread_discretization(phi_num=20)
    plot(vertices, [flatten(faces)])


if __name__ == "__main__":
    main()
