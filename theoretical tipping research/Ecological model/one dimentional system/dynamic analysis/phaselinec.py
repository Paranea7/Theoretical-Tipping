import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define the population model equation
def population_model(t, x, r, K, c):
    growth_term = r * x * (1 - x / K)
    predation_term = c * x ** 2 / (1 + x ** 2)
    return growth_term - predation_term

# Find equilibrium points
def find_equilibria(r, K, c):
    def equation(x):
        return r * x * (1 - x / K) - c * x ** 2 / (1 + x ** 2)

    guesses = [0, 0.1 * K, 0.5 * K, 0.9 * K, 1.2 * K]
    equilibria = []
    for guess in guesses:
        sol = fsolve(equation, guess, full_output=True)
        if sol[2] == 1:
            eq = float(sol[0][0])
            if not any(np.isclose(eq, existing_eq, rtol=1e-3) for existing_eq in equilibria):
                equilibria.append(eq)
    return sorted(equilibria)

# Stability analysis
def stability_analysis(x_eq, r, K, c):
    def derivative(x):
        growth_derivative = r * (1 - 2 * x / K)
        predation_derivative = c * (2 * x / (1 + x ** 2) - 2 * x ** 3 / (1 + x ** 2) ** 2)
        return growth_derivative - predation_derivative

    stability = {}
    for eq in x_eq:
        deriv = derivative(eq)
        if deriv < 0:
            stability[eq] = "Stable"
        elif deriv > 0:
            stability[eq] = "Unstable"
        else:
            stability[eq] = "Requires further analysis"
    return stability

# Time-varying coupling (example: sinusoidal variation)
def time_varying_c(t, c0, delta_c, omega):
    return c0 + delta_c * np.sin(omega * t)

def main():
    # Base parameters
    r = 0.47
    K = 10.0
    c0 = 1.0
    delta_c = 0.2
    omega = 0.5  # frequency for time-varying term

    # Compute equilibria for base case
    equilibria = find_equilibria(r, K, c0)
    stability = stability_analysis(equilibria, r, K, c0)

    print("Equilibrium points and stability (fixed parameters):")
    for eq in equilibria:
        print(f"x = {eq:.4f}: {stability[eq]}")

    # Time range for phase-line related plots
    t_span = [0, 50]
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # 1) Phase line with fixed parameters (baseline for reference)
    plt.figure(figsize=(8, 6))
    x_values = np.linspace(0, max([0.1 * K] + equilibria + [K * 1.2]), 1000)
    dx_dt_fixed = [population_model(0, x, r, K, c0) for x in x_values]

    plt.plot(x_values, dx_dt_fixed, 'b-', linewidth=2, label='dx/dt (fixed c)')
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)

    for eq in equilibria:
        plt.plot(eq, 0, 'ro' if stability[eq] == "Stable" else 'go', markersize=8,
                 label=f"eq x={eq:.2f} ({stability[eq]})")

    plt.xlabel('x (population density)')
    plt.ylabel('dx/dt')
    plt.title('Phase Line (Fixed Parameters)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) Phase line evolution with time-varying c(t)
    # For visualization we sample several time points across [0, 50]
    plt.figure(figsize=(8, 6))
    x_values_tv = np.linspace(0, max([0.1 * K] + equilibria + [K * 1.2]), 1000)
    t_samples = np.linspace(t_span[0], t_span[1], 6)  # 6 samples to illustrate variation

    for t_sample in t_samples:
        c_t = time_varying_c(t_sample, c0, delta_c, omega)
        dxdt_vals = [population_model(0, x, r, K, c_t) for x in x_values_tv]
        plt.plot(x_values_tv, dxdt_vals, label=f't={t_sample:.2f}, c={c_t:.3f}')

    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
    for eq in equilibria:
        plt.plot(eq, population_model(0, eq, r, K, c0), 'kx')
    plt.xlabel('x (population density)')
    plt.ylabel('dx/dt')
    plt.title('Phase-Line-like Plots Under Time-Varying c(t) (samples)')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3) Quantitative metrics over time: number of equilibria and max displacement
    plt.figure(figsize=(8, 4))
    # Track equilibria count over time
    counts = []
    max_disp = []
    for t in t_eval:
        c_t = time_varying_c(t, c0, delta_c, omega)
        eqs = find_equilibria(r, K, c_t)
        counts.append(len(eqs))
        # displacement: max difference between current eqs and something reference (这里简单定义为与固定c0的等效点的差值的最大)
        # 这里为了简单演示，我们取固定参数下的最大等效点作为参照
        max_disp.append(max([abs(eq - e) for eq in eqs for e in equilibria] + [0]))

    plt.plot(t_eval, counts, 'm-', lw=2, label='Number of equilibria dx/dt=0')
    plt.plot(t_eval, max_disp, 'c--', lw=2, label='Max displacement from fixed-c equilibria')
    plt.xlabel('Time t')
    plt.ylabel('Metric')
    plt.title('Evolution of equilibria count and displacement vs time')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()