import numpy as np
import matplotlib.pyplot as plt


# Define the Lotka-Volterra model
def lotka_volterra(H0, P0, r, d, a, b, t_max, dt):
    t = np.arange(0, t_max, dt)
    H = np.zeros(len(t))
    P = np.zeros(len(t))

    H[0] = H0
    P[0] = P0

    for i in range(1, len(t)):
        H[i] = H[i - 1] + (r * H[i - 1] - a * H[i - 1] * P[i - 1]) * dt
        P[i] = P[i - 1] + (b * a * H[i - 1] * P[i - 1] - d * P[i - 1]) * dt

    return t, H, P


# Define the Rosenzweig-MacArthur model
def rosenzweig_macarthur(H0, P0, r, K, alpha, beta, d, t_max, dt):
    t = np.arange(0, t_max, dt)
    H = np.zeros(len(t))
    P = np.zeros(len(t))

    H[0] = H0
    P[0] = P0

    for i in range(1, len(t)):
        H[i] = H[i - 1] + (
                    r * H[i - 1] * (1 - H[i - 1] / K) - (alpha * H[i - 1] * P[i - 1]) / (1 + beta * H[i - 1])) * dt
        P[i] = P[i - 1] + ((beta * H[i - 1] * P[i - 1]) / (1 + beta * H[i - 1]) - d * P[i - 1]) * dt

    return t, H, P


# Set parameters
H0 = 40  # Initial prey population
P0 = 9  # Initial predator population

# Lotka-Volterra parameters
r_LV = 0.1
d_LV = 0.1
a_LV = 0.01
b_LV = 0.01
t_max_LV = 200
dt_LV = 0.1

# Rosenzweig-MacArthur parameters
r_RM = 0.1
K_RM = 100  # Carrying capacity
alpha_RM = 0.01
beta_RM = 0.1
d_RM = 0.1
t_max_RM = 200
dt_RM = 0.1

# Run models
t_LV, H_LV, P_LV = lotka_volterra(H0, P0, r_LV, d_LV, a_LV, b_LV, t_max_LV, dt_LV)
t_RM, H_RM, P_RM = rosenzweig_macarthur(H0, P0, r_RM, K_RM, alpha_RM, beta_RM, d_RM, t_max_RM, dt_RM)

# Plot results
plt.figure(figsize=(14, 12))

# Lotka-Volterra model
plt.subplot(2, 2, 1)
plt.plot(t_LV, H_LV, label='Prey (H)', color='blue')
plt.plot(t_LV, P_LV, label='Predator (P)', color='red')
plt.title('Lotka-Volterra Model Dynamics')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.legend()
plt.grid()

# Rosenzweig-MacArthur model
plt.subplot(2, 2, 2)
plt.plot(t_RM, H_RM, label='Prey (H)', color='blue')
plt.plot(t_RM, P_RM, label='Predator (P)', color='red')
plt.title('Rosenzweig-MacArthur Model Dynamics')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.legend()
plt.grid()

# Phase Portrait for Lotka-Volterra
plt.subplot(2, 2, 3)
plt.plot(H_LV, P_LV, color='purple')
plt.title('Lotka-Volterra Phase Portrait')
plt.xlabel('Prey (H)')
plt.ylabel('Predator (P)')
plt.grid()

# Phase Portrait for Rosenzweig-MacArthur
plt.subplot(2, 2, 4)
plt.plot(H_RM, P_RM, color='green')
plt.title('Rosenzweig-MacArthur Phase Portrait')
plt.xlabel('Prey (H)')
plt.ylabel('Predator (P)')
plt.grid()

plt.tight_layout()
plt.show()