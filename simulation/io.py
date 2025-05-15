import os
import numpy as np
from simulation.constants import n, L
from simulation.utils import minimum_image

# Output Functions
def dump(r, mols, tp, t, outdir="polymer_dumps"):
    # Write current configuration to file in LAMMPS-style format
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/t{t}.dump", "w") as f:
        f.write("ITEM: TIMESTEP\n" + f"{t}\n")
        f.write("ITEM: NUMBER OF ATOMS\n" + f"{n}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        for dim in L:
            f.write(f"0 {dim}\n")
        f.write("ITEM: ATOMS id mol type x y z\n")
        for i in range(n):
            f.write(f"{i} {mols[i]} {tp[i]} {r[i,0]} {r[i,1]} {r[i,2]}\n")

def radius_of_gyration(r, mols):
    # Compute radius of gyration for each chain
    Rg_list = []
    unique_mols = np.unique(mols)
    for m_id in unique_mols:
        r_chain = r[mols == m_id]
        COM = np.mean(r_chain, axis=0)
        Rg2 = np.mean(np.sum((r_chain - COM) ** 2, axis=1))
        Rg_list.append(np.sqrt(Rg2))
    return Rg_list

def mean_squared_displacement(r, r0):
    # Compute mean squared displacement of all particles
    disp = minimum_image(r - r0)
    msd = np.mean(np.sum(disp**2, axis=1))
    return msd