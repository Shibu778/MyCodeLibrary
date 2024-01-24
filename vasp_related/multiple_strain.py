# This scripts can be useful in generating POSCARs applying different strains
# To the initial POSCAR

from pymatgen.core.structure import Structure


# Function to generate the strain
def apply_strain(file_to_read, strain_vec=[0.05, 0, 0]):
    """
    Function to write strained poscar.

    Parameters :
    ============
    file_to_read : str
        File path of a CONTCAR or POSCAR that will be read

    strain_vec : list
        A list of three float representing the strain in x, y, z
        direction in fractions. [0.05, 0, 0] means only 5% strain
        in x direction
    """
    poscar = Structure.from_file(file_to_read)
    tail = (
        str(strain_vec[0])
        + "_"
        + str(strain_vec[1])
        + "_"
        + str(strain_vec[2])
        + ".vasp"
    )
    file_to_write = "./POSCAR_" + tail

    if strain_vec[0] == strain_vec[1] and strain_vec[0] == strain_vec[2]:
        poscar1 = poscar.apply_strain(strain_vec[0], inplace=False)
    else:
        poscar1 = poscar.apply_strain(strain_vec, inplace=False)

    poscar1.to_file(file_to_write)
    print(file_to_write, "is written!")
    return 0


if __name__ == "__main__":
    strains = [
        [0.05, 0, 0],
        [-0.05, 0, 0],
        [0.04, 0, 0],
        [-0.04, 0, 0],
        [0.03, 0, 0],
        [-0.03, 0, 0],
        [0.02, 0, 0],
        [-0.02, 0, 0],
        [0.01, 0, 0],
        [-0.01, 0, 0],
        [0, 0, 0],
    ]

    file_to_read = "./CONTCAR"

    for strain_vec in strains:
        apply_strain(file_to_read, strain_vec=strain_vec)
