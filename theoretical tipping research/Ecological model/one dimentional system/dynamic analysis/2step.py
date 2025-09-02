import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# Define the population model equation
def population_model(t, x, r, K, c):
    """
    Single-species population model:
    dx/dt = r*x*(1 - x/K) - c*x^2/(1 + x^2)
    """
    growth_term = r * x * (1 - x / K)
    predation_term = c * x ** 2 / (1 + x ** 2)
    return growth_term - predation_term

# Find equilibrium points
def find_equilibria(r, K, c):
    """
    Find equilibrium points of the model (dx/dt = 0)
    """
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

# Potential function calculation
def potential_function(x, r, K, c):
    return -r * x**2 / 2 + r * x**3 / (3 * K) + c * (x - np.arctan(x))

# Derivative of the potential with respect to x: dV/dx
def dVdx(x, r, K, c):
    term1 = -r * x
    term2 = (r / K) * x**2
    term3 = c * (1 - 1.0 / (1 + x**2))
    return term1 + term2 + term3

# Time-varying coupling (example: sinusoidal variation)
def time_varying_c(t, c0, delta_c, omega):
    """
    Returns a time-dependent predation/harvesting coefficient.
    """
    return c0 + delta_c * np.sin(omega * t)

# Main program
def main():
    # Base parameters
    r = 0.47
    K = 10.0
    c0 = 1.0
    delta_c = 0.5
    omega = 0.2  # frequency for time-varying term

    # Time range
    t_span = [0, 50]
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Initial conditions
    initial_conditions = [0.1, 1.0, 5.0, 8.0, 12.0, 15.0]

    # Equilibria for base case
    equilibria = find_equilibria(r, K, c0)
    stability = stability_analysis(equilibria, r, K, c0)

    print("Equilibrium points and stability (fixed parameters):")
    for eq in equilibria:
        print(f"x = {eq:.4f}: {stability[eq]}")

    # Prepare figure with multiple subplots
    plt.figure(figsize=(18, 12))  # Large figure for many subplots

    # 1) Population dynamics for several initial conditions (fixed parameters)
    plt.subplot(2, 3, 1)
    for x0 in initial_conditions:
        sol = solve_ivp(
            lambda t, x: population_model(t, x, r, K, c0),
            t_span,
            [x0],
            t_eval=t_eval,
            method='RK45'
        )
        plt.plot(sol.t, sol.y[0], label=f'x(0) = {x0}')
    for eq in equilibria:
        if stability[eq] == "Stable":
            plt.axhline(y=eq, color='r', linestyle='--', linewidth=2, label=f'Stable eq: {eq:.2f}')
        else:
            plt.axhline(y=eq, color='g', linestyle='--', linewidth=2, label=f'Unstable eq: {eq:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Population density x(t)')
    plt.title('Population Dynamics (Fixed Parameters)')
    plt.legend()
    plt.grid(True)

    # 1b) Comparison: time-varying coefficient c(t)
    plt.plot([], [], alpha=0)  # placeholder for legend
    for x0 in initial_conditions:
        sol = solve_ivp(
            lambda t, x: population_model(t, x, r, K, time_varying_c(t, c0, delta_c, omega)),
            t_span,
            [x0],
            t_eval=t_eval,
            method='RK45'
        )
        plt.plot(sol.t, sol.y[0], linestyle='--', color='orange', label=f'x0_timevary {x0}')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout()

    # 2) Phase line (fixed parameters)
    plt.subplot(2, 3, 2)
    x_values = np.linspace(0, max(initial_conditions) * 1.2, 1000)
    dx_dt = [population_model(0, x, r, K, c0) for x in x_values]
    plt.plot(x_values, dx_dt, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=1)
    for eq in equilibria:
        plt.plot(eq, 0, 'ro' if stability[eq] == "Stable" else 'go', markersize=8)
    plt.xlabel('Population density x')
    plt.ylabel('dx/dt')
    plt.title('Phase Line (Fixed Parameters)')
    plt.grid(True)

    # 2b) Phase line with time-varying c(t)
    plt.plot([], [], alpha=0)
    for x0 in initial_conditions:
        sol = solve_ivp(
            lambda t, x: population_model(t, x, r, K, time_varying_c(t, c0, delta_c, omega)),
            t_span,
            [x0],
            t_eval=t_eval,
            method='RK45'
        )
        plt.plot(sol.y[0], [population_model(0, x, r, K, time_varying_c(t, c0, delta_c, omega))
                          for t, x in zip(sol.t, sol.y[0])],
                 linestyle='--', color='orange', alpha=0.3)

    # 3) Zero growth isoclines (fixed parameters)
    plt.subplot(2, 3, 3)
    growth = [r * x * (1 - x / K) for x in x_values]
    predation = [c0 * x ** 2 / (1 + x ** 2) for x in x_values]
    plt.plot(x_values, growth, 'g-', label='Growth term: r*x*(1-x/K)')
    plt.plot(x_values, predation, 'r-', label='Predation term: c*x²/(1+x²)')
    plt.plot(x_values, np.zeros_like(x_values), 'k--')
    for eq in equilibria:
        plt.plot(eq, r * eq * (1 - eq / K), 'bo', markersize=8)
    plt.xlabel('Population density x')
    plt.ylabel('Rate')
    plt.title('Zero Growth Isoclines (Fixed)')
    plt.legend()
    plt.grid(True)

    # 3b) Time-varying comparison: approximate isocline using average c(t)
    plt.plot([], [], alpha=0)

    # 4) Bifurcation diagram (vary c) - fixed vs time-varying
    plt.subplot(2, 3, 4)
    c_values = np.linspace(0, 5, 100)
    equilibria_c_fixed = []
    equilibria_c_tv = []
    for c_val in c_values:
        eqs = find_equilibria(r, K, c_val)
        for eq in eqs:
            equilibria_c_fixed.append((c_val, eq))
        # For time-varying scenario, use the same c_val as baseline
        eqs_tv = find_equilibria(r, K, c_val)
        for eq in eqs_tv:
            equilibria_c_tv.append((c_val, eq))
    if equilibria_c_fixed:
        c_vals, eq_vals = zip(*equilibria_c_fixed)
        plt.scatter(c_vals, eq_vals, s=1, color='blue', label='Fixed c')
    if equilibria_c_tv:
        c_vals_tv, eq_vals_tv = zip(*equilibria_c_tv)
        plt.scatter(c_vals_tv, eq_vals_tv, s=2, color='orange', label='Timed c (example)')
    plt.xlabel('Predation coefficient c')
    plt.ylabel('Equilibrium points')
    plt.title('Bifurcation Diagram (c variation) with Contrast')
    plt.grid(True)
    plt.legend()

    # 5) Potential function V(x) (Fixed)
    plt.subplot(2, 3, 5)
    x_pot = np.linspace(0, 15, 500)
    V_x = [potential_function(x, r, K, c0) for x in x_pot]
    plt.plot(x_pot, V_x)
    plt.xlabel('x')
    plt.ylabel('V(x)')
    plt.title('Potential Function V(x) (Fixed)')
    plt.grid(True)

    # 5b) Time-varying comparison for V(x) (use average c)
    c_avg = c0
    V_x_tv = [potential_function(x, r, K, c_avg) for x in x_pot]
    plt.plot(x_pot, V_x_tv, linestyle='--', color='orange', label='V(x) with c(t) average')
    plt.legend()

    # 6) Phase portrait (x vs. dx/dt) (Fixed)
    plt.subplot(2, 3, 6)
    x_phase = np.linspace(0, 15, 500)
    dxdt_phase = [population_model(0, x, r, K, c0) for x in x_phase]
    plt.plot(x_phase, dxdt_phase)
    plt.xlabel('x')
    plt.ylabel('dx/dt')
    plt.title('Phase Portrait (x vs. dx/dt) Fixed')
    plt.grid(True)

    # 6b) Time-varying comparison in phase portrait (overlay)
    plt.plot([], [], alpha=0)

    plt.tight_layout()
    plt.show()



# Run the main program
if __name__ == "__main__":
    main()