"""
create nanoparticle using wulff construction
"""

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from mpinterfaces import get_struct_from_mp
from mpinterfaces.nanoparticle import Nanoparticle
from pymatgen.io.vasp.inputs import Poscar
from mp_api.client import MPRester

# -----------------------------------
# nanoparticle specifications
# -----------------------------------
# max radius in angstroms
rmax = 4
# surface families to be chopped off
hkl_family = [(1, 1, 0), (1, 0, 4)]
# surfac energies could be in any units, will be normalized
surface_energies = [25, 25]

material_ids = []
file_name = "nanoparticle_fe2o3_r4.xyz"

# -----------------------------------
# initial structure
# -----------------------------------
# caution: set the structure wrt which the the miller indices are
# specified. use your own key (Materials Project API KEY)
# with MPRester("MAPI_KEY") as mpr:
#     docs = mpr.summary.search(material_ids=material_ids)
# structure = docs[0].structure  # Structure of PbS
# structure = get_struct_from_mp("PbS", MAPI_KEY="MAPI_KEY")
structure = Poscar.from_file("./POSCAR_bulk").structure
# primitive ---> conventional cell
sa = SpacegroupAnalyzer(structure)
structure_conventional = sa.get_conventional_standard_structure()
print(structure_conventional)

# -----------------------------------
# create nanoparticle
# -----------------------------------
nanoparticle = Nanoparticle(
    structure_conventional,
    rmax=rmax,
    hkl_family=hkl_family,
    surface_energies=surface_energies,
)
nanoparticle.create()
nanoparticle.to(fmt="xyz", filename=file_name)