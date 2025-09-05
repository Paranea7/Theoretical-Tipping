import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define the population model equation
def population_model(t, x, r_t, K_t, c_t):
    growth_term = r_t * x * (1 - x / K_t)
    predation_term = c_t * x ** 2 / (1 + x ** 2)
    return growth_term - predation_term

# Time-varying parameters (base + sinusoidal perturbations)
def time_varying_params(t, r0, r_A, omega_r, K0, K_A, omega_K, c0, c_A, omega_c):
    r_t = r0 + r_A * np.sin(omega_r * t)
    K_t = K0 + K_A * np.sin(omega_K * t)
    c_t = c0 + c_A * np.sin(omega_c * t)
    return r_t, K_t, c_t

# Find equilibrium points
def find_equilibria(r0, K0, c0, r_A, K_A, c_A, omega_r, omega_K, omega_c, t_guess_span=None):
    def equation(x, t):
        r_t, K_t, c_t = time_varying_params(t, r0, r_A, omega_r, K0, K_A, omega_K, c0, c_A, omega_c)
        return r_t * x * (1 - x / K_t) - c_t * x ** 2 / (1 + x ** 2)

    # Use several time points to collect equilibria across time if desired
    if t_guess_span is None:
        t_guess_span = [0.0]
    equilibria = []
    guesses = [0, 0.1 * K0, 0.5 * K0, 0.9 * K0, 1.2 * K0]
    for t_guess in t_guess_span:
        for guess in guesses:
            sol = fsolve(lambda x: equation(x, t_guess), guess, full_output=True)
            if sol[2] == 1:
                eq = float(sol[0][0])
                if not any(np.isclose(eq, existing_eq, rtol=1e-3) for existing_eq in equilibria):
                    equilibria.append(eq)
    return sorted(equilibria)

# Stability analysis
def stability_analysis(x_eq, r0, K0, c0, r_A, K_A, c_A, omega_r, omega_K, omega_c, t_eval):
    def derivative(x, t):
        r_t, K_t, c_t = time_varying_params(t, r0, r_A, omega_r, K0, K_A, omega_K, c0, c_A, omega_c)
        growth_derivative = r_t * (1 - 2 * x / K_t)
        predation_derivative = c_t * (2 * x / (1 + x ** 2) - 2 * x ** 3 / (1 + x ** 2) ** 2)
        return growth_derivative - predation_derivative

    stability = {}
    for eq in x_eq:
        # 取一个代表性时间点的导数值来判断稳定性
        deriv = derivative(eq, t_eval)
        if deriv < 0:
            stability[eq] = "Stable"
        elif deriv > 0:
            stability[eq] = "Unstable"
        else:
            stability[eq] = "Requires further analysis"
    return stability

# Time ranges and parameters
def main():
    # 基线参数（用于对比）
    r0 = 0.47
    K0 = 10.0
    c0 = 1.0

    # 振幅与频率（默认 C_A=K_A=0，以实现基线对比；若要查看时变效应，可把振幅设为非零）
    r_A = 0.05      # r 振幅（基线为0）
    K_A = 1.0      # K 振幅（基线为0）
    c_A = 0.2      # c 振幅（基线为0）

    omega_r = 0.2
    omega_K = 0.0
    omega_c = 0.2

    # 1) 基线下的等值点（r=0.47, K=10, c=1，且不随时间变化）
    equilibria_baseline = find_equilibria(r0, K0, c0, r_A, K_A, c_A, omega_r, omega_K, omega_c, t_guess_span=[0.0, 5.0, 10.0])
    stability_baseline = stability_analysis(equilibria_baseline, r0, K0, c0, r_A, K_A, c_A, omega_r, omega_K, omega_c, t_eval=0.0)

    print("Baseline (r=0.47, K=10, c=1) Equilibria and stability:")
    for eq in equilibria_baseline:
        print(f"x = {eq:.4f}: {stability_baseline.get(eq, 'Unknown')}")

    # 2) 时变情形的阶段性观测
    # 时间范围
    t_span = [0, 200]
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # 2.1) Phase line with baseline parameters for reference
    plt.figure(figsize=(8, 6))
    x_values = np.linspace(0, max([0.1 * K0] + equilibria_baseline + [K0 * 1.2]), 1000)
    dx_dt_fixed = [population_model(0, x, r0, K0, c0) for x in x_values]

    plt.plot(x_values, dx_dt_fixed, 'b-', linewidth=2, label='dx/dt (baseline fixed c, r, K)')
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)

    for eq in equilibria_baseline:
        plt.plot(eq, 0, 'ro', markersize=8, label=f"eq x={eq:.2f}")

    plt.xlabel('x (population density)')
    plt.ylabel('dx/dt')
    plt.title('Phase Line: Baseline Parameters (r=0.47, K=10, c=1)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2.2) Phase line evolution with time-varying parameters
    plt.figure(figsize=(8, 6))
    x_values_tv = np.linspace(0, max([0.1 * K0] + equilibria_baseline + [K0 * 1.2]), 1000)
    t_samples = np.linspace(t_span[0], t_span[1], 6)

    for t_sample in t_samples:
        r_t, K_t, c_t = time_varying_params(t_sample, r0, r_A, omega_r, K0, K_A, omega_K, c0, c_A, omega_c)
        dxdt_vals = [population_model(t_sample, x, r_t, K_t, c_t) for x in x_values_tv]
        plt.plot(x_values_tv, dxdt_vals, label=f't={t_sample:.2f}, r={r_t:.3f}, K={K_t:.3f}, c={c_t:.3f}')

    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
    for eq in equilibria_baseline:
        plt.plot(eq, population_model(t_span[0], eq, r0, K0, c0), 'kx')
    plt.xlabel('x (population density)')
    plt.ylabel('dx/dt')
    plt.title('Phase-Line with Time-Varying Parameters (samples)')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3) Quantitative metrics over time: number of equilibria and max displacement
    plt.figure(figsize=(8, 4))
    counts = []
    max_disp = []
    for t in t_eval:
        r_t, K_t, c_t = time_varying_params(t, r0, r_A, omega_r, K0, K_A, omega_K, c0, c_A, omega_c)
        eqs = find_equilibria(r0, K0, c0, r_A, K_A, c_A, omega_r, omega_K, omega_c, t_guess_span=[t])
        counts.append(len(eqs))
        max_disp.append(max([abs(eq - e) for eq in eqs for e in equilibria_baseline] + [0]))

    plt.plot(t_eval, counts, 'm-', lw=2, label='Number of equilibria dx/dt=0 (time-varying)')
    plt.plot(t_eval, max_disp, 'c--', lw=2, label='Max displacement from baseline equilibria')
    plt.xlabel('Time t')
    plt.ylabel('Metric')
    plt.title('Evolution of equilibria count and displacement vs time (baseline vs time-varying)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()