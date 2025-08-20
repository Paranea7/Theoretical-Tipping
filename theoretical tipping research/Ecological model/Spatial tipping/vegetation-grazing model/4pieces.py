import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def f(V, r, V_c, c):
    """Reaction term function"""
    return r * V * (1 - V / V_c) - c * V ** 2 / (V ** 2 + 1)


def f_prime(V, r, V_c, c):
    """Derivative of reaction term"""
    return r * (1 - 2 * V / V_c) - 2 * c * V / (V ** 2 + 1) ** 2


def checkerboard_eq(V, r, V_c, c, D, h):
    """Checkerboard equilibrium equations: V11=V22=Va, V12=V21=Vb"""
    Va, Vb = V
    beta = D / h ** 2
    eq1 = f(Va, r, V_c, c) + 2 * beta * (Vb - Va)
    eq2 = f(Vb, r, V_c, c) + 2 * beta * (Va - Vb)
    return [eq1, eq2]


def row_sync_eq(V, r, V_c, c, D, h):
    """Row-synchronized equilibrium equations: V11=V12=Vr, V21=V22=Vc"""
    Vr, Vc = V
    beta = D / h ** 2
    eq1 = f(Vr, r, V_c, c) + beta * (Vc - Vr)  # For V11 and V12
    eq2 = f(Vc, r, V_c, c) + beta * (Vr - Vc)  # For V21 and V22
    return [eq1, eq2]


# Parameter settings
r, V_c, c, D, h = 1.0, 10.0, 2.0, 0.1, 1.0

# Create figure with 3x3 subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# 1. Plot reaction term function f(V) for all patterns
V_range = np.linspace(0, 10, 1000)
f_values = f(V_range, r, V_c, c)

# Plot f(V) for all patterns
for i in range(3):
    axes[0, i].plot(V_range, f_values, 'b-', linewidth=2, label='f(V)')
    axes[0, i].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[0, i].set_xlabel('V')
    axes[0, i].set_ylabel('f(V)')
    axes[0, i].grid(True, alpha=0.3)
    axes[0, i].legend()

# 2. Find and mark synchronous equilibria
zero_crossings = np.where(np.diff(np.sign(f_values)))[0]
sync_equilibria = []

for i in zero_crossings:
    V_guess = V_range[i]
    V_eq = fsolve(lambda V: f(V, r, V_c, c), V_guess)[0]
    if V_eq >= 0 and V_eq <= 10:  # Ensure within reasonable range
        sync_equilibria.append(V_eq)

print("Synchronous equilibria:", sync_equilibria)

# 3. Find checkerboard equilibria
initial_guesses = [(0.5, 8.0), (0.8, 7.5), (7.5, 0.8)]
checker_equilibria = []

for guess in initial_guesses:
    sol = fsolve(lambda V: checkerboard_eq(V, r, V_c, c, D, h), guess)
    residual = checkerboard_eq(sol, r, V_c, c, D, h)
    if np.allclose(residual, [0, 0], atol=1e-6):
        checker_equilibria.append(sol)

print("Checkerboard equilibria:", checker_equilibria)

# 4. Find row-synchronized equilibria
row_sync_equilibria = []

for guess in initial_guesses:
    sol = fsolve(lambda V: row_sync_eq(V, r, V_c, c, D, h), guess)
    residual = row_sync_eq(sol, r, V_c, c, D, h)
    if np.allclose(residual, [0, 0], atol=1e-6):
        row_sync_equilibria.append(sol)

print("Row-synchronized equilibria:", row_sync_equilibria)

# 5. Plot f(V)-V for fully synchronized pattern
axes[0, 0].set_title('Fully Synchronized Pattern: f(V) vs V')

# Mark synchronous equilibrium points
for i, V_eq in enumerate(sync_equilibria):
    axes[0, 0].plot(V_eq, 0, 'ro', markersize=8)
    axes[0, 0].annotate(f'V_{i + 1} = {V_eq:.3f}',
                        xy=(V_eq, 0),
                        xytext=(V_eq + 0.5, 10 if i % 2 == 0 else -10),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10)

# 6. Plot f(V)-V for checkerboard pattern
axes[0, 1].set_title('Checkerboard Pattern: f(V) vs V')

if checker_equilibria:
    for Va, Vb in checker_equilibria:
        # Mark Va and Vb points on f(V) curve
        axes[0, 1].plot(Va, f(Va, r, V_c, c), 'go', markersize=8)
        axes[0, 1].plot(Vb, f(Vb, r, V_c, c), 'go', markersize=8)

        # Add vertical lines to show the equilibrium values
        axes[0, 1].axvline(Va, color='green', linestyle='--', alpha=0.7)
        axes[0, 1].axvline(Vb, color='green', linestyle='--', alpha=0.7)

        # Add labels
        axes[0, 1].text(Va, axes[0, 1].get_ylim()[0] * 0.9, f'V_a={Va:.3f}',
                        ha='center', color='green')
        axes[0, 1].text(Vb, axes[0, 1].get_ylim()[0] * 0.9, f'V_b={Vb:.3f}',
                        ha='center', color='green')

        # Add the diffusion-modified reaction terms
        beta = D / h ** 2
        modified_fa = f(Va, r, V_c, c) + 2 * beta * (Vb - Va)
        modified_fb = f(Vb, r, V_c, c) + 2 * beta * (Va - Vb)

        axes[0, 1].plot(Va, modified_fa, 'mo', markersize=8)
        axes[0, 1].plot(Vb, modified_fb, 'mo', markersize=8)

        axes[0, 1].text(Va, modified_fa, f'mod f(V_a)={modified_fa:.3f}',
                        ha='center', va='bottom', color='magenta')
        axes[0, 1].text(Vb, modified_fb, f'mod f(V_b)={modified_fb:.3f}',
                        ha='center', va='bottom', color='magenta')

# 7. Plot f(V)-V for row-synchronized pattern
axes[0, 2].set_title('Row-Synchronized Pattern: f(V) vs V')

if row_sync_equilibria:
    for Vr, Vc in row_sync_equilibria:
        # Mark Vr and Vc points on f(V) curve
        axes[0, 2].plot(Vr, f(Vr, r, V_c, c), 'co', markersize=8)
        axes[0, 2].plot(Vc, f(Vc, r, V_c, c), 'co', markersize=8)

        # Add vertical lines to show the equilibrium values
        axes[0, 2].axvline(Vr, color='cyan', linestyle='--', alpha=0.7)
        axes[0, 2].axvline(Vc, color='cyan', linestyle='--', alpha=0.7)

        # Add labels
        axes[0, 2].text(Vr, axes[0, 2].get_ylim()[0] * 0.9, f'V_r={Vr:.3f}',
                        ha='center', color='cyan')
        axes[0, 2].text(Vc, axes[0, 2].get_ylim()[0] * 0.9, f'V_c={Vc:.3f}',
                        ha='center', color='cyan')

        # Add the diffusion-modified reaction terms
        beta = D / h ** 2
        modified_fr = f(Vr, r, V_c, c) + beta * (Vc - Vr)
        modified_fc = f(Vc, r, V_c, c) + beta * (Vr - Vc)

        axes[0, 2].plot(Vr, modified_fr, 'yo', markersize=8)
        axes[0, 2].plot(Vc, modified_fc, 'yo', markersize=8)

        axes[0, 2].text(Vr, modified_fr, f'mod f(V_r)={modified_fr:.3f}',
                        ha='center', va='bottom', color='yellow')
        axes[0, 2].text(Vc, modified_fc, f'mod f(V_c)={modified_fc:.3f}',
                        ha='center', va='bottom', color='yellow')

# 8. Plot pattern visualizations
patterns = [
    ("Fully Synchronized", sync_equilibria[2] if len(sync_equilibria) > 2 else None, 'viridis'),
    ("Checkerboard", checker_equilibria[0] if checker_equilibria else None, 'plasma'),
    ("Row-Synchronized", row_sync_equilibria[0] if row_sync_equilibria else None, 'inferno')
]

for i, (pattern_name, pattern_data, cmap_name) in enumerate(patterns):
    if pattern_data is None:
        axes[1, i].text(0.5, 0.5, f'No {pattern_name.lower()} equilibria found',
                        ha='center', va='center', transform=axes[1, i].transAxes)
        axes[1, i].set_title(f'{pattern_name} Pattern')
        continue

    if pattern_name == "Fully Synchronized":
        # Create 2x2 grid with all values equal
        grid = np.array([[pattern_data, pattern_data], [pattern_data, pattern_data]])
        title = f'{pattern_name} Pattern: V = {pattern_data:.3f}'
    elif pattern_name == "Checkerboard":
        Va, Vb = pattern_data
        grid = np.array([[Va, Vb], [Vb, Va]])
        title = f'{pattern_name} Pattern: V_a={Va:.3f}, V_b={Vb:.3f}'
    else:  # Row-Synchronized
        Vr, Vc = pattern_data
        grid = np.array([[Vr, Vr], [Vc, Vc]])
        title = f'{pattern_name} Pattern: V_r={Vr:.3f}, V_c={Vc:.3f}'

    # Create color mapping
    norm = mcolors.Normalize(vmin=np.min(grid), vmax=np.max(grid))
    cmap = plt.get_cmap(cmap_name)

    # Plot grid
    im = axes[1, i].imshow(grid, cmap=cmap, norm=norm, origin='upper')

    # Add value labels
    for row in range(2):
        for col in range(2):
            text = axes[1, i].text(col, row, f'{grid[row, col]:.3f}',
                                   ha="center", va="center", color="w", fontweight='bold')

    axes[1, i].set_title(title)
    axes[1, i].set_xticks([0, 1])
    axes[1, i].set_yticks([0, 1])
    axes[1, i].set_xticklabels(['1', '2'])
    axes[1, i].set_yticklabels(['1', '2'])
    axes[1, i].set_xlabel('x')
    axes[1, i].set_ylabel('y')

    # Add colorbar
    plt.colorbar(im, ax=axes[1, i], label='V value')

# 9. Plot phase plane analysis for each pattern
phase_planes = [
    ("Fully Synchronized", sync_equilibria, 'red'),
    ("Checkerboard", checker_equilibria, 'green'),
    ("Row-Synchronized", row_sync_equilibria, 'cyan')
]

for i, (pattern_name, equilibria, color) in enumerate(phase_planes):
    # Create a simple phase plane visualization
    if pattern_name == "Fully Synchronized":
        # For fully synchronized, we can plot the derivative f'(V)
        df_values = f_prime(V_range, r, V_c, c)
        axes[2, i].plot(V_range, df_values, 'b-', linewidth=2)
        axes[2, i].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[2, i].set_xlabel('V')
        axes[2, i].set_ylabel("f'(V)")
        axes[2, i].set_title(f'{pattern_name}: Stability Analysis')
        axes[2, i].grid(True, alpha=0.3)

        # Mark equilibrium points
        for V_eq in equilibria:
            stability = "stable" if f_prime(V_eq, r, V_c, c) < 0 else "unstable"
            axes[2, i].plot(V_eq, 0, 'o', color=color, markersize=8)
            axes[2, i].text(V_eq, axes[2, i].get_ylim()[1] * 0.9,
                            f'V={V_eq:.2f}\n({stability})',
                            ha='center', color=color)

    else:
        # For other patterns, create a simple 2D visualization
        if not equilibria:
            axes[2, i].text(0.5, 0.5, f'No {pattern_name.lower()} equilibria',
                            ha='center', va='center', transform=axes[2, i].transAxes)
            axes[2, i].set_title(f'{pattern_name} Pattern')
            continue

        # Create a grid of values
        if pattern_name == "Checkerboard":
            Va, Vb = equilibria[0]
            x = [Va, Vb]
            y = [f(Va, r, V_c, c), f(Vb, r, V_c, c)]
            labels = ['V_a', 'V_b']
        else:  # Row-Synchronized
            Vr, Vc = equilibria[0]
            x = [Vr, Vc]
            y = [f(Vr, r, V_c, c), f(Vc, r, V_c, c)]
            labels = ['V_r', 'V_c']

        axes[2, i].plot(V_range, f_values, 'b-', linewidth=2, label='f(V)')
        axes[2, i].axhline(0, color='k', linestyle='--', alpha=0.5)

        # Plot the equilibrium points
        axes[2, i].plot(x, y, 'o', color=color, markersize=8)
        for j, (x_val, y_val, label) in enumerate(zip(x, y, labels)):
            axes[2, i].text(x_val, y_val, f'{label}={x_val:.2f}',
                            ha='center', va='bottom', color=color)

        axes[2, i].set_xlabel('V')
        axes[2, i].set_ylabel('f(V)')
        axes[2, i].set_title(f'{pattern_name}: f(V) at Equilibrium Points')
        axes[2, i].grid(True, alpha=0.3)
        axes[2, i].legend()

plt.suptitle('Analysis of 2×2 Grid Reaction-Diffusion System with f(V)-V Plots', fontsize=16)
plt.savefig('reaction_diffusion_fV_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Output detailed analysis results
print("\n=== Detailed Analysis Results ===")
print(f"Parameters: r={r}, V_c={V_c}, c={c}, D={D}, h={h}")

print("\nSynchronous equilibria:")
for i, V_eq in enumerate(sync_equilibria):
    stability = "stable" if f_prime(V_eq, r, V_c, c) < 0 else "unstable"
    print(f"  V_{i + 1} = {V_eq:.4f} ({stability})")

print("\nCheckerboard equilibria:")
if checker_equilibria:
    for i, (Va, Vb) in enumerate(checker_equilibria):
        print(f"  Solution {i + 1}: V_a = {Va:.4f}, V_b = {Vb:.4f}")

        # Check if equations are satisfied
        residual = checkerboard_eq([Va, Vb], r, V_c, c, D, h)
        print(f"    Residual: [{residual[0]:.6f}, {residual[1]:.6f}]")
else:
    print("  No checkerboard equilibria found")

print("\nRow-synchronized equilibria:")
if row_sync_equilibria:
    for i, (Vr, Vc) in enumerate(row_sync_equilibria):
        print(f"  Solution {i + 1}: V_r = {Vr:.4f}, V_c = {Vc:.4f}")

        # Check if equations are satisfied
        residual = row_sync_eq([Vr, Vc], r, V_c, c, D, h)
        print(f"    Residual: [{residual[0]:.6f}, {residual[1]:.6f}]")
else:
    print("  No row-synchronized equilibria found")