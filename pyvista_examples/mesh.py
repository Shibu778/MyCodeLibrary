import numpy as np
import pyvista as pv

# Generate points cloud
points = np.random.rand(100, 3)
mesh = pv.PolyData(points)
mesh.plot(point_size=10, style="points", color="tan")


# Most meshes have connectivity between the points
from pyvista import examples

mesh = examples.load_hexbeam()
cpos = [(6.20, 3.00, 7.50), (0.16, 0.13, 2.65), (-0.28, 0.94, -0.21)]

pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True, color="white")
pl.add_points(mesh.points, color="red", point_size=20)
pl.camera_position = cpos
pl.show()

# Meshes can have triangulated surface
mesh = examples.download_bunny_coarse()

pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True, color="white")
pl.add_points(mesh.points, color="red", point_size=20)
pl.camera_position = [(0.02, 0.30, 0.73), (0.02, 0.03, -0.022), (-0.03, 0.94, -0.34)]
pl.show()

# Cell
mesh = examples.load_hexbeam()

pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True, color="white")
pl.add_points(mesh.points, color="red", point_size=20)

single_cell = mesh.extract_cells(mesh.n_cells - 1)
pl.add_mesh(single_cell, color="pink", edge_color="blue", line_width=5, show_edges=True)

pl.camera_position = [(6.20, 3.00, 7.50), (0.16, 0.13, 2.65), (-0.28, 0.94, -0.21)]
pl.show()

# Attributes
# point_data, cell_data, field_data
mesh.point_data["my point values"] = np.arange(mesh.n_points)
mesh.plot(scalars="my point values", cpos=cpos, show_edges=True)

mesh.cell_data["my cell values"] = np.arange(mesh.n_cells)
mesh.plot(scalars="my cell values", cpos=cpos, show_edges=True)


# Cell data and point data comparison
import pyvista as pv
from pyvista import examples

uni = examples.load_uniform()

pl = pv.Plotter(shape=(1, 2), border=False)
pl.add_mesh(uni, scalars="Spatial Point Data", show_edges=True)
pl.subplot(0, 1)
pl.add_mesh(uni, scalars="Spatial Cell Data", show_edges=True)
pl.show()

cube = pv.Cube()
cube.cell_data["myscalars"] = range(6)

other_cube = cube.copy()
other_cube.point_data["myscalars"] = range(8)

pl = pv.Plotter(shape=(1, 2), border_width=1)
pl.add_mesh(cube, cmap="coolwarm")
pl.subplot(0, 1)
pl.add_mesh(other_cube, cmap="coolwarm")
pl.show()
