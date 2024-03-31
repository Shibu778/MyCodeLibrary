# This script is used to simulate a force field
# inside a composite made up of polymer composite
# reinforced with Fe2O3 nanoparticle

# Plan
# 1. Keep everything in the first quadrant

# Imports
import magpylib as magpy
import numpy as np

# Geometry of the system
## Geometry of strip
x_width_s = 0.07  # 7cm
y_width_s = 0.02  # 2 cm
z_width_s = 0.001  # 0.1 cm

## Starting Position of strip
x_start_s = 0
y_start_s = 0.05
z_start_s = 0.05

# Find the griding in sample

x_grid = np.linspace(x_start_s, x_start_s + x_width_s, num=100)
y_grid = np.linspace(y_start_s, y_start_s + y_width_s, num=100)
z_grid = np.linspace(z_start_s, z_start_s + z_width_s, num=10)
s_grid = np.array(np.meshgrid(x_grid, y_grid, z_grid))

## Geometry of magnet
x_width_m = 0.05  # 4 cm
y_width_m = 0.04  # 5 cm
z_width_m = 0.06  # 6 cm

## Starting Position of magnet
x_start_m = 0.03 + x_width_m / 2
y_start_m = 0 + y_width_m / 2
z_start_m = 0 + z_width_m / 2


# Calculate the Magnetic Field inside the composite structure

## Magnetic moment of the commercial magnet
m_source = (0, 0, 1)  # In Tesla

# define the magnet
magnet = magpy.magnet.Cuboid(
    polarization=m_source,  # in SI Units (T)
    dimension=(x_width_m, y_width_m, z_width_m),  # in SI Units (m)
    position=(x_start_m, y_start_m, z_start_m),
)
print(magnet.position)

# magpy.show(magnet)
s_grid = np.swapaxes(s_grid, 0, 3)
B_s = magpy.getB(magnet, observers=s_grid)

print(B_s.shape)
print(B_s[0][0][0])  # Magnetic Field at coordinate
print(s_grid[0][0][0])
print(magpy.getB(magnet, observers=s_grid[0][0][0]))

# I can use three different backend
# backend = "plotly" or "matplotlib" or "pyvista"
magpy.show(magnet)
