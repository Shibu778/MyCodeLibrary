import numpy as np

import pyvista as pv
from pyvista import examples

# A vtkStructuredGrid - but could be any mesh type
mesh = examples.download_carotid()

mesh_g = mesh.compute_derivative(scalars="vectors")


def gradients_to_dict(arr):
    """A helper method to label the gradients into a dictionary."""
    keys = np.array(
        [
            "du/dx",
            "du/dy",
            "du/dz",
            "dv/dx",
            "dv/dy",
            "dv/dz",
            "dw/dx",
            "dw/dy",
            "dw/dz",
        ]
    )
    keys = keys.reshape((3, 3))[:, : arr.shape[1]].ravel()
    return dict(zip(keys, mesh_g["gradient"].T))


gradients = gradients_to_dict(mesh_g["gradient"])

mesh_g.point_data.update(gradients)

keys = np.array(list(gradients.keys())).reshape(3, 3)

# p = pv.Plotter(shape=keys.shape)
# for (i, j), name in np.ndenumerate(keys):
#     p.subplot(i, j)
#     p.add_mesh(mesh_g.contour(scalars=name), scalars=name, opacity=0.75)
#     p.add_mesh(mesh_g.outline(), color="k")
# p.link_views()
# p.view_isometric()
# p.show()

# Scalar field
mesh_g = mesh.compute_derivative(scalars="scalars")

gradients = gradients_to_dict(mesh_g["gradient"])

mesh_g.point_data.update(gradients)

keys = np.array(list(gradients.keys())).reshape(1, 3)

p = pv.Plotter(shape=keys.shape)

for (i, j), name in np.ndenumerate(keys):
    name = keys[i, j]
    p.subplot(i, j)
    p.add_mesh(mesh_g.contour(scalars=name), scalars=name, opacity=0.75)
    p.add_mesh(mesh_g.outline(), color="k")
p.link_views()
p.view_isometric()
p.show()
