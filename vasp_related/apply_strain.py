# This scripts apply strain to a structure and write a new POSCAR
from pymatgen.core.structure import Structure

file_to_read = "./CONTCAR"

# + sign : elongate, - sign : compress, list element : percentage in fraction
strain_vec = [0.05, 0, 0]

poscar = Structure.from_file(file_to_read)
tail = (
    str(strain_vec[0]) + "_" + str(strain_vec[1]) + "_" + str(strain_vec[2]) + ".vasp"
)
file_to_write = "./POSCAR_" + tail

if strain_vec[0] == strain_vec[1] and strain_vec[0] == strain_vec[2]:
    poscar1 = poscar.apply_strain(strain_vec[0], inplace=False)
else:
    poscar1 = poscar.apply_strain(strain_vec, inplace=False)

poscar1.to_file(file_to_write)
