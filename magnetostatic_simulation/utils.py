import numpy as np
from matplotlib import pyplot as plt


def dipole(m, r, r0):
    """Calculate a field in point r created by a dipole moment m located in r0.
    Spatial components are the outermost axis of r and returned B.

    Reference : https://michal.rawlik.pl/2015/03/12/magnetic-dipole-in-python/

    Parameters:
    ===========
    m : list or np.array
        x, y and z component of magnetic moment

    r : list or np.array
        x, y, z coordinate of point or points where we need the value of B

    r0 : list or np.array
        x, y, z coordinate of point where the magnetic dipole lies

    Returns:
    ========
    B : np.array
        x, y and z component of magnetic field vector at given r

    """
    # we use np.subtract to allow r and r0 to be a python lists, not only np.array
    R = np.subtract(np.transpose(r), r0).T

    # assume that the spatial components of r are the outermost axis
    norm_R = np.sqrt(np.einsum("i...,i...", R, R))

    # calculate the dot product only for the outermost axis,
    # that is the spatial components
    m_dot_R = np.tensordot(m, R, axes=1)

    # tensordot with axes=0 does a general outer product - we want no sum
    B = 3 * m_dot_R * R / norm_R**5 - np.tensordot(m, 1 / norm_R**3, axes=0)

    # include the physical constant
    B *= 1e-7

    return B


# Use
# 3D
# dipole(m=[1, 2, 3], r=[1, 1, 2], r0=[0, 0, 0])

# 2D

# X = np.linspace(-1, 1)
# Y = np.linspace(-1, 1)

# Bx, By = dipole(m=[0, 0.1], r=np.meshgrid(X, Y), r0=[-0.2,0.1])

# plt.figure(figsize=(8, 8))
# plt.streamplot(X, Y, Bx, By)
# plt.margins(0, 0)
