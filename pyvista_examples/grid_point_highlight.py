import pyvista as pv

# Create a simple mesh (you can replace this with your own mesh)
mesh = pv.Sphere(radius=1.0)

# Assume you have a list of indices to highlight (e.g., [10, 20, 30])
indices_to_highlight = [10, 20, 30]

# Add labels to the specified points
plotter = pv.Plotter()
plotter.add_mesh(mesh, show_edges=True, color="lightblue")  # Display the mesh
plotter.add_point_labels(
    mesh.points[indices_to_highlight],  # Points to label
    mesh.points[indices_to_highlight].tolist(),  # Labels
    point_size=20,
    font_size=36,
)  # Customize label appearance
plotter.show()  # Show the plot
