import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

COLORS = plt.get_cmap('tab20').colors


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


def create_arrows(ax: mpl.axes.Axes, normals: np.ndarray, expand=1.05):
    points = normals * expand
    ax.quiver(points[:, 0], points[:, 1], points[:, 2], normals[:, 0],
              normals[:, 1], normals[:, 2], length=0.2, normalize=False, color=COLORS[6], linewidths=3.0)


def create_samples(normal=np.array([0, 1, 0]), std=0.01, size=50):
    # create 2D samples
    # import ipdb; ipdb.set_trace()
    np.random.seed(4)
    cov = np.identity(2) * std
    samples_2d = np.random.multivariate_normal([0.0, 0.0], cov, size=size)
    samples_3d = np.append(samples_2d, np.ones((samples_2d.shape[0], 1)), 1)
    samples_3d, _ = normalized(samples_3d)
    # create rotation matrix
    default_normal = np.array([0, 0, 1])
    rm, _,= Rotation.align_vectors([default_normal], [normal])
    normals_3d = rm.apply(samples_3d)
    # TODO normalize
    return normals_3d
