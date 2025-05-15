import numpy as np
from simulation.constants import kb, n

# Temperature Rescaling
def rescaleT(v, mm, T_target):
    # Rescales velocities to match the target temperature
    KE = 0.5 * np.sum(mm[:, None] * v**2)
    T_current = (2 / (3 * n)) * KE / kb
    scale_factor = np.sqrt(T_target / T_current)
    return v * scale_factor