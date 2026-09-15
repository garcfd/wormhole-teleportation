

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Define the Geodesic Equations of Motion
def wormhole_geodesics(tau, state, b0, L):
    """
    Defines the differential equations for a particle in a Morris-Thorne wormhole.
    state = [l, phi, p_l]
    l: proper radial distance
    phi: angular coordinate
    p_l: radial momentum (dl/dtau)
    """
    l, phi, p_l = state

    # dl/dtau
    dl_dtau = p_l

    # dphi/dtau
    dphi_dtau = L / (b0**2 + l**2)

    # dp_l/dtau (Radial acceleration due to geometry and angular momentum)
    # Derived directly from Christoffel symbols of the metric
    dp_l_dtau = (L**2 * l) / (b0**2 + l**2)**2

    return [dl_dtau, dphi_dtau, dp_l_dtau]

# 2. Simulation Parameters
b0 = 1.0       # Radius of the wormhole throat
L = 0.5        # Angular momentum (Keep low so it doesn't bounce off the centripetal barrier)

# Initial conditions: [l0, phi0, p_l0]
# Starting far away in Universe A (l = -10) moving toward the throat (p_l = 1.2)
initial_state = [-10.0, 0.0, 1.2]

# Time span for the simulation (Proper time tau)
tau_span = (0, 20)
tau_eval = np.linspace(tau_span[0], tau_span[1], 1000)

# 3. Run the Numerical Integration
solution = solve_ivp(
    wormhole_geodesics,
    tau_span,
    initial_state,
    args=(b0, L),
    t_eval=tau_eval,
    method='RK45'
)

# Extract results
tau = solution.t
l_vals = solution.y[0]
phi_vals = solution.y[1]

# 4. Map Coordinates for Visualization
# Convert the intrinsic coordinates (l, phi) into an embedding space (r, phi)
# r is the circumferential radius: r = sqrt(b0^2 + l^2)
r_vals = np.sqrt(b0**2 + l_vals**2)

# Convert to Cartesian (X, Y) coordinates for a standard 2D top-down trajectory plot
# We use sign(l) * r to cleanly separate Universe A (-) from Universe B (+) visually
X = np.sign(l_vals) * r_vals * np.cos(phi_vals)
Y = np.sign(l_vals) * r_vals * np.sin(phi_vals)

# 5. Plotting the Results
plt.figure(figsize=(12, 5))

# Plot 1: Radial Position vs Proper Time (The Transition)
plt.subplot(1, 2, 1)
plt.plot(tau, l_vals, color='indigo', lw=2.5, label='Object Path')
plt.axhline(0, color='crimson', linestyle='--', alpha=0.7, label='Wormhole Throat (l=0)')
plt.fill_between(tau, l_vals, 0, where=(l_vals < 0), color='blue', alpha=0.05, label='Universe A')
plt.fill_between(tau, l_vals, 0, where=(l_vals > 0), color='green', alpha=0.05, label='Universe B')
plt.title('Radial Distance ($l$) vs Proper Time ($\\tau$)')
plt.xlabel('Proper Time ($\\tau$)')
plt.ylabel('Proper Radial Distance ($l$)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: 2D Spatial Trajectory Mapping
plt.subplot(1, 2, 2)
# Draw the Throat
throat_phi = np.linspace(0, 2*np.pi, 200)
plt.plot(b0 * np.cos(throat_phi), b0 * np.sin(throat_phi), color='crimson', lw=2, linestyle='--', label='Throat Entrance')

# Draw the Trajectory
plt.plot(X, Y, color='black', lw=2, label='Trajectory')
plt.scatter(X[0], Y[0], color='blue', s=100, zorder=5, label='Start (Universe A)')
plt.scatter(X[-1], Y[-1], color='green', s=100, zorder=5, label='End (Universe B)')

plt.title('Top-Down Trajectory Projection')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

