from pyvista import examples
import pyvista as pv

dataset = examples.download_saddle_surface()
# dataset
dataset.plot(color="tan")

dataset = examples.download_frog()
dataset.plot(volume=True)

# Download from https://github.com/naucoin/VTKData/blob/master/Data/ironProt.vtk
dataset = pv.read("./ironProt.vtk")
dataset.plot(volume=True)

dataset = examples.download_pine_roots()
dataset.plot()

dataset = examples.download_bolt_nut()
pl = pv.Plotter()
_ = pl.add_volume(
    dataset,
    cmap="coolwarm",
    opacity="sigmoid_5",
    show_scalar_bar=False,
)
pl.camera_position = [(194.6, -141.8, 182.0), (34.5, 61.0, 32.5), (-0.229, 0.45, 0.86)]
pl.show()

# Sample vtk files : https://github.com/pyvista/vtk-data/tree/master/Data
# Sample STL files : https://www.amtekcompany.com/teaching-resources/stl-files/
#
