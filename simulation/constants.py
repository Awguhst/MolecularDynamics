import numpy as np

# Simulation Parameters
n_monomers = 100                             # Number of monomers per polymer chain
n_PE_chains = 1                             # Number of polyethylene chains
n_PS_chains = 1                             # Number of polystyrene chains
n_chains = n_PE_chains + n_PS_chains        # Total number of polymer chains
n = n_monomers * n_chains                   # Total number of particles (atoms/monomers)
D = 3                                       # Dimensionality of the simulation (3D)
L = np.full(D, 30.0)                        # Simulation box dimensions in each direction

dt = 0.0005                                 # Time step size (ps)
T0 = 250                                    # Target temperature in Kelvin

kb = 0.8314459920816467                     # Boltzmann constant in MD units
NA = 6.0221409e+26                          # Avogadro's number (used for unit conversion)
ech = 1.60217662E-19                        # Elementary charge in Coulombs
kc = 8.9875517923E9 * NA * 1e30 * ech**2 / 1e24  # Coulomb constant in MD units

# === Particle Properties ===
mass_types = [12.0, 104.0]                  # Masses: PE monomer (C) and PS monomer (C+phenyl)

sig = [[3.93, 4.215], [4.215, 4.50]]        # Lennard-Jones σ parameters for PE and PS
eps = [[0.091, np.sqrt(0.091 * 0.295)],     # ε parameters for LJ interaction
       [np.sqrt(0.091 * 0.295), 0.295]]

bond_length = 1.54                          # Bond length between monomers (Å)
K_bond = 70000.0                            # Bond spring constant

angle_params = {                            # Angle equilibrium and stiffness for PE and PS
    0: (112.0 * np.pi / 180.0, 62.0),       # PE: angle in radians, force constant
    1: (120.0 * np.pi / 180.0, 150.0),      # PS: more rigid due to aromatic ring
}

cutoff_LJ = 2.5                             # Lennard-Jones interaction cutoff distance
skin = 0.3                                  # Skin for neighbor list buffer
neighbor_update_interval = 10               # How frequently neighbor list is updated