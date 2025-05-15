import numpy as np
from simulation.constants import L, cutoff_LJ, skin, n

# PBC Utilities
def apply_PBC(r):
    # Applies periodic boundary conditions to positions
    return r % L

def minimum_image(rij):
    # Minimum image convention to find the shortest vector between particles under PBC
    return rij - L * np.round(rij / L)

# Neighbor List Construction
def build_neighbor_list(r, cutoff=cutoff_LJ + skin):
    # Builds a neighbor list for efficient force computation
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            rij = minimum_image(r[i] - r[j])
            dist = np.linalg.norm(rij)
            if dist < cutoff:
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors