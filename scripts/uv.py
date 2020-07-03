import math
import time
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

from .normals import create_samples, create_arrows
from scipy.spatial.ckdtree import cKDTree


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


# def spread_discretization(phi_num=20):
#     phis = np.linspace(0, math.pi, phi_num + 1, endpoint=True)
#     vh = VertexHolder()
#     faces_all = []
#     cells_all = []
#     cells_bounds_all = []
#     for i in range(1, phi_num):
#         phi = phis[i]
#         nc = 2 * phi_num * math.sin(phi)
#         nc = int(np.round(nc))
#         thetas = np.linspace(0, 2 * np.pi, nc, endpoint=False)

#         phi_top = (phis[i - 1] + phis[i]) / 2.0
#         phi_bottom = (phis[i + 1] + phis[i]) / 2.0
#         theta_step = (thetas[1] - thetas[0]) / 2.0
#         faces = []
#         cells = []
#         cells_bounds = []
#         for theta in thetas:
#             theta_left = theta - theta_step
#             theta_right = theta + theta_step

#             p_center = polar_to_xyz(phi, theta)
#             cells.append(p_center)
#             cells_bounds.append([phi_top, phi_bottom, theta_left, theta_right])

#             p1 = polar_to_xyz(phi_top, theta_left)
#             p2 = polar_to_xyz(phi_top, theta_right)
#             p3 = polar_to_xyz(phi_bottom, theta_right)
#             p4 = polar_to_xyz(phi_bottom, theta_left)

#             # so that there is a curve between p3 to p4
#             bunch_of_thetas = np.linspace(theta_right, theta_left, num=10, endpoint=False)
#             bunch_of_points = [polar_to_xyz(phi_bottom, b_t) for b_t in bunch_of_thetas[1:]]

#             p_indices = [vh.add_vertex(p) for p in [p1, p2, p3, *bunch_of_points, p4]]
#             faces.append(p_indices)
#         cells_all.append(cells)
#         cells_bounds_all.append(cells_bounds)
#         faces_all.append(np.array(faces))
    
#     cells_all.insert(0, [[0, 0, 1]])
#     cells_all.append([[0, 0, -1]])

#     phi_bottom = (phis[0] + phis[1]) / 2.0
#     phi_top = (phis[-2] + phis[-1]) / 2.0
#     cells_bounds_all.insert(0, [[0, phi_bottom, 0, 2*math.pi]])
#     cells_bounds_all.append([[phi_top,math.pi,0, 2*math.pi]])

#     return np.row_stack(vh.all_vertices), faces_all, cells_all, cells_bounds_all, phis

def spread_discretization(phi_num=20):
    phis = np.linspace(0, math.pi, phi_num, endpoint=True)
    vh = VertexHolder()
    faces_all = []
    cells_all = []
    cells_bounds_all = []
    # print(phis)
    for i in range(1, phi_num -2):
        phi = phis[i]
        nc = 2 * phi_num * math.sin(phi)
        nc = int(np.round(nc))
        thetas = np.linspace(0, 2 * np.pi, nc, endpoint=False)

        phi_top = phis[i]
        phi_bottom = phis[i+1]
        theta_step = (thetas[1] - thetas[0]) / 2.0
        faces = []
        cells = []
        cells_bounds = []
        for theta in thetas:
            theta_left = theta - theta_step
            theta_right = theta + theta_step

            p_center = polar_to_xyz((phi_top + phi_bottom) / 2.0, theta)
            cells.append(p_center)
            cells_bounds.append([phi_top, phi_bottom, theta_left, theta_right])

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
        cells_bounds_all.append(cells_bounds)
        faces_all.append(np.array(faces))
        # print(cells_bounds)
        # import ipdb; ipdb.set_trace()
    
    cells_all.insert(0, [[0, 0, 1]])
    cells_all.append([[0, 0, -1]])

    phi_bottom = (phis[0] + phis[1]) / 2.0
    phi_top = (phis[-2] + phis[-1]) / 2.0
    cells_bounds_all.insert(0, [[0, phi_bottom, 0, 2*math.pi]])
    cells_bounds_all.append([[phi_top,math.pi,0, 2*math.pi]])

    return np.row_stack(vh.all_vertices), faces_all, cells_all, cells_bounds_all, phis


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

def plt_show():
    '''Text-blocking version of plt.show()
    Use this instead of plt.show()'''
    plt.draw()
    plt.pause(0.001)
    input("Press enter to continue...")

def plot(vertices, faces_list):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')

    for faces in faces_list:
        draw_faces(vertices, faces, ax)
    set_axes_equal(ax)

    samples_z = create_samples_and_arrow(ax, normal=np.array([0, 0, 1]), size=100)
    samples_y = create_samples_and_arrow(ax, normal=np.array([1, 0, 0]), size=100)

    # Hide axes ticks
    plt.axis('off')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # get correct view
    ax.view_init(elev=0, azim=180)
    ax.dist = 6.4
    plt_show()

    # next one
    ax.view_init(elev=90, azim=180)
    ax.dist = 6.4
    plt_show()


    plt.close()

    return [samples_z, samples_y]


    # fig.close()

def plot_histogram_grid(samples, grid:np.ndarray):
    all_samples = np.concatenate(samples, axis=0)
    grid_flat = grid.reshape((grid.shape[0] * grid.shape[1], 3))
    tree = cKDTree(grid_flat, leafsize=8)

    # import ipdb; ipdb.set_trace()
    _, indices = tree.query(all_samples, k=1)

    image = np.zeros(grid.shape[:2], dtype=np.uint8)
    image_flat = image.reshape((grid.shape[0] * grid.shape[1], ))
    np.add.at(image_flat, indices, 1)
    # image_flat[indices] = 255
    image = np.swapaxes(image, 0, 1)
    plt.imshow(image)
    plt_show()
    plt.close()

def strips_to_grid(strip_cells):
    rows = len(strip_cells)
    columns =  np.max([len(cells) for cells in strip_cells])
    print(rows, columns)
    image = np.zeros((rows, columns), dtype=np.uint8)

    grid_strip = np.zeros((rows, columns, 3), dtype='f8')

    for cells in strip_cells:
        print(len(cells))


def cell_values_to_strips(cells_list, cells_values):

    idx = 0
    cells_values_list = []
    for i in range(len(cells_list)):
        cells = cells_list[i]
        num_cells = len(cells)
        values_list = cells_values[idx:idx+num_cells]
        cells_values_list.append(values_list)
        idx += num_cells
    return cells_values_list


def get_strip_values(samples, cells):

    all_samples = np.concatenate(samples, axis=0)
    cells_np = np.array(flatten(cells))
    
    cells_values = np.zeros((cells_np.shape[0], ), dtype=np.uint8)
    tree = cKDTree(cells_np, leafsize=8)
    _, indices = tree.query(all_samples, k=1)
    np.add.at(cells_values, indices, 1)

    cells_value_list = cell_values_to_strips(cells, cells_values)

    return cells_value_list

def plot_histogram_strips(samples, cells_list, cells_bounds_list, phis):
    strip_cell_values = get_strip_values(samples, cells_list)

    fig =plt.figure(figsize=(4,4))
    ax = fig.gca()
    height = 0.16534698
    height = phis[1] - phis[0]
    total_height = 0.0
    for i in range(len(cells_list)):
        cell_bounds = cells_bounds_list[i]
        for j in range(len(cell_bounds)):
            y1, y2, x1, x2 = cell_bounds[j]
            width = np.abs(x2 - x1)
            # print(y1, y2)
            # import ipdb; ipdb.set_trace()
            # print(total_height)
            rect_patch = Rectangle((x1, total_height), width, height, ec='k')
            ax.add_patch(rect_patch)
        total_height += height
            
    ax.set_xlabel('theta')
    ax.set_ylabel('phi')
    ax.set_xlim([-0.4, 6.5])
    ax.set_ylim([-0.4, 3.6])
    plt.show()
    # import ipdb; ipdb.set_trace()




    strips_to_grid(cells)


def main():
    # UV Sphere
    # lats, longs = uv_discretization(phi=20, theta=20)
    # grid, shape = create_xyz_grid(longs, lats)
    # vertices, faces = create_cells(grid)
    # samples = plot(vertices, [faces])
    # plot_histogram_grid(samples, grid)
    # 
    vertices, faces, cells, cells_bounds, phis = spread_discretization(phi_num=20)
    samples = plot(vertices, [flatten(faces)])
    plot_histogram_strips(samples, cells, cells_bounds, phis)


if __name__ == "__main__":
    main()
