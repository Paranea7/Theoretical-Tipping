import numpy as np

# Defining parameters
N = 2  # Number of sites (e.g., for a 2-level system, can be adjusted)
dt = 0.01  # Time step
num_steps = 1000  # Number of time steps

# Parameters for Hamiltonian
Omega_1 = 1.0  # Coupling strength for first driver
Omega_2 = 1.0  # Coupling strength for second driver
Delta_1 = 0.5  # Energy offset for r1
Delta_2 = 0.5  # Energy offset for r2
Gamma_1 = 1.0  # Decay rate for r1
Gamma_2 = 1.0  # Decay rate for r2
V_ij = 0.1  # Interaction strength

# Initialize density matrix (2x2 for a two-level system for example)
rho = np.array([[1, 0], [0, 0]], dtype=complex)  # Pure state |0><0|


# Define Hamiltonian H
def create_hamiltonian():
    H = np.zeros((N, N), dtype=complex)

    for i in range(N):
        H[i, i] += Delta_1 if i == 0 else Delta_2  # Energy offsets

    # Adding coupling terms (example)
    H[0, 1] += Omega_1
    H[1, 0] += Omega_1

    # Interaction terms can be added here if necessary
    # Example for a simple two-site interaction
    H[0, 0] += 0.5 * V_ij  # Just an illustrative term, modify for your case
    H[1, 1] += 0.5 * V_ij

    return H


H = create_hamiltonian()


# Define Lindblad operators
def L_r1(rho):
    n_i_r1 = np.diag([0, 1])  # Particle number operator for r1
    sig_r1_g = np.array([[0, 1], [0, 0]], dtype=complex)  # Transition operator
    return (Gamma_1 / 2) * (2 * (sig_r1_g @ rho @ sig_r1_g.conj().T) - np.trace(n_i_r1 @ rho) * rho)


def L_r2(rho):
    n_i_r2 = np.diag([1, 0])  # Particle number operator for r2
    sig_r2_r1 = np.array([[0, 1], [0, 0]], dtype=complex)  # Transition operator
    return (Gamma_2 / 2) * (2 * (sig_r2_r1 @ rho @ sig_r2_r1.conj().T) - np.trace(n_i_r2 @ rho) * rho)


# Time evolution loop
for step in range(num_steps):
    # Compute the Hamiltonian term
    H_term = 1j * (H @ rho - rho @ H)

    # Update the density matrix according to the equation
    rho += (H_term + L_r1(rho) + L_r2(rho)) * dt

    # Normalize the density matrix to ensure it's still a valid state.
    rho = rho / np.trace(rho)

# Print final density matrix
print("Final Density Matrix:")
print(rho)