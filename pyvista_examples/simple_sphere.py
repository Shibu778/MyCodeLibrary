import pyvista as pv

# Creating a simple sphere
mesh = pv.Sphere()

# Print number of points
print("Number of points : ", mesh.points.shape)

# Print number of faces
print("Number of faces : ", mesh.faces.shape)

# Plot with ege
mesh.plot(background="w", show_edges=True)

# Plot metallic sphere
mesh.plot(background="w", pbr=True, metallic=1.0)

# Give scalar value to the sphere in z direction
scalars = mesh.points[:, 2]
mesh.plot(background="w", scalars=scalars)
