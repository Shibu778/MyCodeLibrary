# This script finds out the neighbouring atom list of all atoms

from monty.serialization import loadfn
from pydefect.util.structure_tools import Distances
import json

filename="./supercell_info.json"
supercell = loadfn(filename)

structure = supercell.structure
neighbour_data = {}
for i in range(len(structure)):
    coordination_ind = Distances(structure, structure.frac_coords[i]).coordination().neighboring_atom_indices
    neighbour_data[i] = coordination_ind

# print(neighbour_data)
filename = "./neighbor_data.json"
with open(filename, "w") as f:
    json.dump(neighbour_data, f)
