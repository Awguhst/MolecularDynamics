import numpy as np
from simulation.constants import sig, eps, cutoff_LJ, n
from simulation.utils import minimum_image

# Lennard-Jones Force Calculation
def LJ_force(tp, r, neighbor_list):
    F = np.zeros_like(r)  # Initialize force array

    for i in range(n):
        for j in neighbor_list[i]:
            if j <= i:
                continue  # Avoid double counting (i-j and j-i are symmetric)

            # Compute distance vector using minimum image convention
            rij = minimum_image(r[i] - r[j])
            dist = np.linalg.norm(rij)

            # Skip if distance is too small (avoids singularity) or outside cutoff
            if dist < 1e-8 or dist > cutoff_LJ:
                continue

            # Get interaction parameters based on particle types
            type_i, type_j = tp[i], tp[j]
            σ = sig[type_i][type_j]
            ε = eps[type_i][type_j]

            # Compute Lennard-Jones force magnitude
            r6 = (σ / dist) ** 6
            r12 = r6 ** 2
            fmag = 24 * ε * (2 * r12 - r6) / dist**2

            # Convert magnitude to force vector
            fvec = fmag * rij / dist

            # Apply Newton's third law: action = -reaction
            F[i] += fvec
            F[j] -= fvec

    return F
    
def bond_force(bnd, r):
    F = np.zeros_like(r)  # Initialize force array

    for i, j, r0, k in bnd:
        # Ensure indices are integers (if stored as floats)
        i, j = int(i), int(j)

        # Compute bond vector and apply minimum image convention
        rij = minimum_image(r[i] - r[j])
        dist = np.linalg.norm(rij)

        # Harmonic force: F = -k * (r - r0)
        f_magnitude = k * (dist - r0)
        f_direction = rij / dist

        # Apply force to both atoms (Newton's third law)
        F[i] -= f_magnitude * f_direction
        F[j] += f_magnitude * f_direction

    return F

def angle_force(angs, r):
    F = np.zeros_like(r)  # Initialize force array

    for i, j, k, th0, Kth in angs:
        i, j, k = int(i), int(j), int(k)  # Ensure indices are integers

        # Vectors from vertex atom j to atoms i and k
        rij = minimum_image(r[i] - r[j])
        rkj = minimum_image(r[k] - r[j])
        norm_rij = np.linalg.norm(rij)
        norm_rkj = np.linalg.norm(rkj)

        # Avoid divide-by-zero errors
        if norm_rij == 0 or norm_rkj == 0:
            continue

        # Unit vectors
        u = rij / norm_rij
        v = rkj / norm_rkj

        # Angle between u and v
        cos_theta = np.clip(np.dot(u, v), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        dtheta = theta - th0

        # Avoid numerical instability for small angles
        sin_theta = np.sqrt(1 - cos_theta**2)
        if sin_theta < 1e-6:
            continue

        # Force magnitude based on harmonic potential
        fmag = -Kth * dtheta / sin_theta

        # Gradients of angle w.r.t. positions
        du = (v - u * cos_theta) / norm_rij
        dv = (u - v * cos_theta) / norm_rkj

        # Forces on each atom
        Fi = fmag * du
        Fk = fmag * dv
        Fj = -Fi - Fk  # Force conservation

        F[i] += Fi
        F[j] += Fj
        F[k] += Fk

    return F
    
# Total Force
def total_force(tp, bnd, angs, r, neighbor_list):
    # Sum of all force contributions
    return (
        LJ_force(tp, r, neighbor_list)
        + bond_force(bnd, r)
        + angle_force(angs, r)
    )