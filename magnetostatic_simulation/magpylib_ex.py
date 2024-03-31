# This code shows the use of magpie lib
import magpylib as magpy

# Create a Cuboid magnet with sides 1,2 and 3 cm respectively, and a polarization
# of 1000 mT pointing in x-direction.
cube = magpy.magnet.Cuboid(
    polarization=(1, 0, 0),  # in SI Units (T)
    dimension=(0.01, 0.02, 0.03),  # in SI Units (m)
)

# By default, the magnet position is (0,0,0) and its orientation is the unit
# rotation (given by a scipy rotation object), which corresponds to magnet sided
# parallel to global coordinate axes.
print(cube.position)  # --> [0. 0. 0.]
print(cube.orientation.as_rotvec())  # --> [0. 0. 0.]

# Manipulate object position and orientation through the respective attributes,
# or by using the powerful `move` and `rotate` methods.
cube.move((0, 0, -0.02))  # in SI Units (m)
cube.rotate_from_angax(angle=45, axis="z")
print(cube.position)  # --> [0. 0. -0.02]
print(cube.orientation.as_rotvec(degrees=True))  # --> [0. 0. 45.]

# Compute the magnetic B-field in units of T at a set of observer positions. Magpylib
# makes use of vectorized computation. Hand over all field computation instances,
# e.g. different observer positions, at one function call. Avoid Python loops !!!
observers = [(0, 0, 0), (0.01, 0, 0), (0.02, 0, 0)]  # in SI Units (m)
B = magpy.getB(cube, observers)
print(B.round(2))  # --> [[-0.09 -0.09  0.  ]
#                         [ 0.   -0.04  0.08]
#                         [ 0.02 -0.01  0.03]]  # in SI Units (T)

# Sensors are observer objects that can have their own position and orientation.
# Compute the H-field in units of A/m.
sensor = magpy.Sensor(position=(0, 0, 0))
sensor.rotate_from_angax(angle=45, axis=(1, 1, 1))
H = magpy.getH(cube, sensor)
print(H.round())  # --> [-94537. -35642. -14085.]  # in SI Units (A/m)

# Position and orientation attributes of Magpylib objects can be vectors of
# multiple positions/orientations referred to as "paths". When computing the
# magnetic field of an object with a path, it is computed at every path index.
cube.position = [(0, 0, -0.02), (1, 0, -0.02), (2, 0, -0.02)]  # in SI Units (m)
B = cube.getB(sensor)
print(B.round(2))  # --> [[-0.12 -0.04 -0.02]
#                         [ 0.   -0.    0.  ]
#                         [ 0.   -0.    0.  ]] # in SI Units (T)

# When several objects are involved and things are getting complex, make use of
# the `show` function to view your system through Matplotlib, Plotly or Pyvista backends.
magpy.show(cube, sensor)
