# This script creates complex defects, e.g. XSi1VC1 and other
# such configuration

from monty.serialization import loadfn
from pymatgen.core.structure import Structure
supercell_info = loadfn("./supercell_info.json")
supercell_struct = supercell_info.structure
import os

def substitute(initial_structure, ind, dopant):
    """
    Substitute an atom at index (ind) atom by dopant.
    """
    structure = initial_structure.as_dict()
    structure['sites'][ind]['label'] = dopant
    structure['sites'][ind]['species'][0]['element'] = dopant

    final_structure = Structure.from_dict(structure)
    return final_structure

def remove(initial_structure, ind):
    """
    Remove the atom at index ind.
    """
    structure = initial_structure.as_dict()
    structure['sites'].pop(ind)
    final_structure = Structure.from_dict(structure)
    return final_structure

def create_divacancies(initial_structure, ind1, ind2):
    structure = initial_structure.as_dict()
    ind1_site = structure["sites"][ind1]
    ind2_site = structure["sites"][ind2]
    structure.remove(ind1_site)
    structure.remove(ind2_site)
    final_struct = Structure.from_dict(structure)
    return final_structure 

def make_X_A_Y_B(initial_structure, X, ind1, Y, ind2):
    """
    Makes X_A_Y_B defect where X is the dopant and V is vacancy, ind1 and ind2 are
    corresponding indices in pristine supercell.
    """
    if X == "Va":
        sub_struct = substitute(initial_structure, ind2, Y)
        final_struct = remove(sub_struct, ind1)
    elif Y == "Va":
        sub_struct = substitute(initial_structure, ind1, X)
        final_struct = remove(sub_struct, ind2)
    elif X == "Va" and Y == "Va":

    elif X != "Va" and Y != "Va":
        sub_struct = substitute(initial_structure, ind1, X)
        final_struct = substitute(sub_struct, ind2, Y)
    return final_struct

def gen_dir(path):
    """
    Generates directories for charged complex defects.
    """
    os.makedirs(path, exist_ok=True)
    print("{path} directory generated!!")


# [Si, C] site
site_info = {'hh' : [[0, 64],['Si1', 'C1']], 
            'hk' : [[0, 112], ['Si1', 'C2']], 
            'kh': [[32, 64],['Si2', 'C1']], 
            'kk': [[32, 96], ['Si2', 'C2']]}
charges = [-2, -1, 0, 1, 2]
root_path = "./"
for defect in ['Ge', 'Sn', 'Pb', 'Va']:
    for conf in site_info:
        # XSiVaC
        X = defect
        Y = "Va"
        A = site_info[conf][1][0]
        ind1 = site_info[conf][0][0]
        B = site_info[conf][1][1]
        ind2 = site_info[conf][0][1]
        defect_name1 = X + "_" + A + "_" + Y + "_" + B
        defect_stuct1 = make_X_A_Y_B(supercell_struct, X, ind1, Y, ind2)
        for chg in charges:
            path = root_path + defect_name1 + "/" + defect_name1 + "_" + str(chg)
            gen_dir(path)
            defect_stuct1.to(path + "/POSCAR")
        # 
