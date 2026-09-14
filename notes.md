# Physics Documentation: Relativistic Traversable Wormhole Simulation

This document outlines the theoretical physics, differential geometry, and coordinate mapping framework used to simulate a 3D object passing through a traversable wormhole.

---
## 1. The Spacetime Metric (The Foundation)
## 2. Intrinsic Coordinate Transformation
## 3. Geodesic Equations of Motion (The Trajectory)
## 4. Cartesian Visual Mapping & Cinematic Decoupling


## 1. The Spacetime Metric (The Foundation)
The simulation is built upon the **Morris-Thorne Metric** (1988), which describes a static, spherically symmetric, traversable wormhole in General Relativity. In standard spherical coordinates $(t, r, \theta, \phi)$, the spacetime interval $ds^2$ is defined as:

$$ds^2 = -c^2 e^{2\Phi(r)} dt^2 + \frac{dr^2}{1 - \frac{b(r)}{r}} + r^2 (d\theta^2 + \sin^2\theta \, d\phi^2)$$

Where:
* **$ds$**: The spacetime interval.
* **$\Phi(r)$**: The *redshift function*, which determines gravitational time dilation. In this model, we set $\Phi(r) = 0$ to simulate a **zero-tidal-force wormhole** (clocks tick at the same rate everywhere, making it safe for human travel).
* **$b(r)$**: The *shape function*, defining the spatial geometry of the wormhole. We utilize $b(r) = \frac{b_0^2}{r}$, where **$b_0$** is the physical radius of the wormhole throat.

---

## 2. Intrinsic Coordinate Transformation
To smoothly connect two separate universes (Universe A and Universe B) without creating a mathematical singularity or boundary edge at the throat, we switch from the circumferential radius $r$ to the **proper radial distance ($l$)**:

$$r = \sqrt{b_0^2 + l^2}$$

This transformation creates a continuous coordinate line mapping the entire journey:
* **$l \to -\infty$**: Asymptotically flat region of **Universe A** (The Entrance).
* **$l = 0$**: The **Wormhole Throat**, where the circumferential radius reaches its absolute physical minimum ($r = b_0$).
* **$l \to +\infty$**: Asymptotically flat region of **Universe B** (The Exit).

---

## 3. Geodesic Equations of Motion (The Trajectory)
The path of an unpowered object falling freely through curved spacetime is a **geodesic**, governed by the differential equation:

$$\frac{d^2 x^\mu}{d\tau^2} + \Gamma^\mu_{\alpha\beta} \frac{dx^\alpha}{d\tau} \frac{dx^\beta}{d\tau} = 0$$

By evaluating the Christoffel symbols ($\Gamma^\mu_{\alpha\beta}$) for our specific metric on a 2D orbital plane ($\theta = \pi/2$), the motion simplifies to a system of coupled **Ordinary Differential Equations (ODEs)** solved over proper time ($\tau$):

$$\frac{dl}{d\tau} = p_l$$

$$\frac{dp_l}{d\tau} = \frac{L^2 \cdot l}{(b_0^2 + l^2)^2}$$

$$\frac{d\phi}{d\tau} = \frac{L}{b_0^2 + l^2}$$

Where:
* **$\tau$**: The proper time experienced by the object itself.
* **$L$**: The conserved orbital angular momentum per unit mass.
* **$p_l$**: The radial momentum. The term $\frac{L^2 \cdot l}{(b_0^2 + l^2)^2}$ acts as a geometric centripetal barrier pushing back against the object if it approaches at an angle.

---

## 4. Cartesian Visual Mapping & Cinematic Decoupling
Because standard 3D rendering engines operate in flat Euclidean Cartesian space $(X, Y, Z)$ rather than curved spacetime, the cross-sectional dimensions (the $Y$ and $Z$ axes perpendicular to the axis of travel) are warped frame-by-frame based on the ratio of curved space to flat space:

$$\text{Base Scale Factor} = \frac{r(l)}{\vert{}l\vert{} + b_0} = \frac{\sqrt{b_0^2 + l^2}}{\vert{}l\vert{} + b_0}$$

To allow for custom cinematic control, the physics metric is decoupled in code using two independent tuning variables:
1. **$S_{\text{shrink}}$ (Longitudinal Compression Exponent)**: Distorts the $X$-axis positioning non-linearly to simulate severe pancake compression along the direction of travel as the object nears $l=0$.
2. **$E_{\text{expand}}$ (Lateral Flare Multiplier)**: Dynamically scales the cross-sectional $Y$ and $Z$ dimensions at the throat interface, allowing independent amplification of the trumpet-like ballooning effect.

