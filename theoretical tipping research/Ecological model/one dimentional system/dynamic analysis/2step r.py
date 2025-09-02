import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# Model: dx/dt = r*x*(1 - x/K) - c*x^2/(1+x^2)
def population_model(t, x, r, K, c):
    growth_term = r * x * (1 - x / K)
    predation_term = c * x ** 2 / (1 + x ** 2)
    return growth_term - predation_term

# Equilibria (dx/dt = 0)
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
            stability[eq] = "Indeterminate"
    return stability

# Potential function and derivative (optional diagnostics)
def potential_function(x, r, K, c):
    return -r * x**2 / 2 + r * x**3 / (3 * K) + c * (x - np.arctan(x))

def dVdx(x, r, K, c):
    term1 = -r * x
    term2 = (r / K) * x**2
    term3 = c * (1 - 1.0 / (1 + x**2))
    return term1 + term2 + term3

# Time-varying c(t)
def time_varying_c(t, c0, delta_c, omega):
    return c0 + delta_c * np.sin(omega * t)

# Main
def main():
    # Fixed baseline parameters
    K = 10.0
    c0 = 1.0
    delta_c = 0.5
    omega = 0.2  # frequency for time-varying term

    r = 0.47  # keep as fixed parameter for this visualization
    t_span = [0, 50]
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    initial_conditions = [0.1, 1.0, 5.0, 8.0, 12.0, 15.0]

    # Equilibria for fixed c0
    equilibria = find_equilibria(r, K, c0)
    stability = stability_analysis(equilibria, r, K, c0)

    print("Equilibria (fixed c0) and stability:")
    for eq in equilibria:
        print(f"x = {eq:.4f}: {stability[eq]}")

    plt.figure(figsize=(18, 12))

    # 1) Population dynamics: fixed c0
    plt.subplot(2, 3, 1)
    for x0 in initial_conditions:
        sol = solve_ivp(
            lambda t, x: population_model(t, x, r, K, c0),
            t_span,
            [x0],
            t_eval=t_eval,
            method='RK45'
        )
        plt.plot(sol.t, sol.y[0], label=f'x0={x0}')
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.title('Population Dynamics (fixed c0)')
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)

    # 1b) Time-varying c(t) dynamics
    plt.subplot(2, 3, 4)  # place in a separate panel for clarity
    for x0 in initial_conditions:
        sol = solve_ivp(
            lambda t, x: population_model(t, x, r, K, time_varying_c(t, c0, delta_c, omega)),
            t_span,
            [x0],
            t_eval=t_eval,
            method='RK45'
        )
        plt.plot(sol.t, sol.y[0], linestyle='--', label=f'x0={x0}')
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.title('Population Dynamics (time-varying c(t))')
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)

    # 2) Phase line for fixed c0
    plt.subplot(2, 3, 2)
    x_values = np.linspace(0, max(initial_conditions) * 1.2, 1000)
    dx_dt = [population_model(0, x, r, K, c0) for x in x_values]
    plt.plot(x_values, dx_dt, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--')
    for eq in equilibria:
        plt.plot(eq, 0, 'ro' if stability[eq] == "Stable" else 'go', markersize=8)
    plt.xlabel('x')
    plt.ylabel('dx/dt')
    plt.title('Phase Line (fixed c0)')
    plt.grid(True)

    # 3) Zero growth isoclines (fixed c0)
    plt.subplot(2, 3, 3)
    x_values = np.linspace(0, max(initial_conditions) * 1.2, 1000)
    growth = [r * x * (1 - x / K) for x in x_values]
    predation = [c0 * x ** 2 / (1 + x ** 2) for x in x_values]
    plt.plot(x_values, growth, 'g-', label='Growth term')
    plt.plot(x_values, predation, 'r-', label='Predation term')
    plt.plot(x_values, np.zeros_like(x_values), 'k--')
    for eq in equilibria:
        plt.plot(eq, r * eq * (1 - eq / K), 'bo', markersize=8)
    plt.xlabel('x')
    plt.ylabel('Rate')
    plt.title('Zero Growth Isoclines (fixed c0)')
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True)

    # 4) Bifurcation-like diagram: sweep r (optional)
    plt.subplot(2, 3, 5)
    r_values = np.linspace(0.1, 1.0, 200)
    bif_points = []
    for rv in r_values:
        eqs = find_equilibria(rv, K, c0)
        for eq in eqs:
            bif_points.append((rv, eq, stability_analysis([eq], rv, K, c0)[eq]))
    if bif_points:
        rs, xs, sts = zip(*bif_points)
        colors = ['blue' if s == 'Stable' else 'orange' for s in sts]
        plt.scatter(rs, xs, c=colors, s=8)
    plt.xlabel('r')
    plt.ylabel('equilibrium x')
    plt.title('Bifurcation (x_eq vs r) [fixed c0]')
    plt.grid(True)

    # 5) Time-varying comparison: potential and dV/dt
    plt.subplot(2, 3, 6)
    x_vals = np.linspace(0, 15, 500)
    V_fixed = [potential_function(x, r, K, c0) for x in x_vals]
    plt.plot(x_vals, V_fixed, label='V(x) @ fixed r')
    # Use a representative r (same r as above) and c varying in a time-averaged sense
    c_avg = c0  # could also use (c0 + time_average(delta_c)) if desired
    V_tv = [potential_function(x, r, K, c_avg) for x in x_vals]
    plt.plot(x_vals, V_tv, linestyle='--', color='orange', label='V(x) with c(t) avg')
    plt.xlabel('x')
    plt.ylabel('V(x)')
    plt.title('Potential Function Comparison')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()