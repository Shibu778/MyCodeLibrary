import numpy as np

import pyvista as pv

ni, nj, nk = 4, 5, 6
si, sj, sk = 20, 10, 1

xcorn = np.arange(0, (ni + 1) * si, si)  # [ 0, 20, 40, 60, 80]
xcorn = np.repeat(xcorn, 2)  # [ 0,  0, 20, 20, 40, 40, 60, 60, 80, 80]
xcorn = xcorn[1:-1]  # [ 0, 20, 20, 40, 40, 60, 60, 80]
xcorn = np.tile(xcorn, 4 * nj * nk)  # repeats xcorn 4*nj*nk times

ycorn = np.arange(0, (nj + 1) * sj, sj)  # [ 0, 10, 20, 30, 40, 50]
ycorn = np.repeat(ycorn, 2)  # [ 0,  0, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50]
ycorn = ycorn[1:-1]  # [ 0, 10, 10, 20, 20, 30, 30, 40, 40, 50]
ycorn = np.tile(ycorn, (2 * ni, 2 * nk))  # (8, 120) dimensional array
ycorn = np.transpose(ycorn)  # (120, 8) dimensional array
ycorn = ycorn.flatten()  # (960,) dimensional array

zcorn = np.arange(0, (nk + 1) * sk, sk)  # [0, 1, 2, 3, 4, 5, 6]
zcorn = np.repeat(zcorn, 2)  # [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
zcorn = zcorn[1:-1]  # [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]
zcorn = np.repeat(zcorn, (4 * ni * nj))  # (960,)

corners = np.stack((xcorn, ycorn, zcorn))  # (3, 960)
corners = corners.transpose()  #  (960, 3)

dims = np.asarray((ni, nj, nk)) + 1  # [5, 6, 7]
grid = pv.ExplicitStructuredGrid(
    dims, corners
)  # Generates a Grid using dims an corners
grid = grid.compute_connectivity()  # Compute the faces connectivity flags array.
grid.plot(show_edges=True)
