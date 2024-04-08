import pyvista as pv
from pyvista import examples

mesh = examples.download_notch_displacement()

dargs = dict(
    scalars="Nodal Displacement",
    cmap="jet",
    show_scalar_bar=False,
)

pl = pv.Plotter(shape=(2, 2))
pl.subplot(0, 0)
pl.add_mesh(mesh, **dargs)
pl.add_text("Normalized Displacement", color="k")
pl.subplot(0, 1)
pl.add_mesh(mesh.copy(), component=0, **dargs)
pl.add_text("X Displacement", color="k")
pl.subplot(1, 0)
pl.add_mesh(mesh.copy(), component=1, **dargs)
pl.add_text("Y Displacement", color="k")
pl.subplot(1, 1)
pl.add_mesh(mesh.copy(), component=2, **dargs)
pl.add_text("Z Displacement", color="k")
pl.link_views()
pl.camera_position = "iso"
pl.background_color = "white"
pl.show()
