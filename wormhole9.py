import numpy as np
from stl import mesh
import os
import pyvista as pv

# ============================================================
# Shared wormhole parameters
# ============================================================
b0 = 1.5  # Radius of both wormhole mouths (spherical throat)
warp_strength = 3.0  # Amplification of the warpage effect (1.0 = original)
W1_center = np.array([ 0.0, 0.0, 0.0])  # Entrance Mouth
W2_center = np.array([10.0, 0.0, 0.0])  # Exit Mouth

# ============================================================
# Part 1: Car transit animation
# Each TRIANGLE is snapped wholly to one mouth (wormhole6 behaviour),
# so the car never stretches across the gap between the two spheres.
# ============================================================

# 1. Load the car mesh
stl_path = 'car3.stl'
if not os.path.exists(stl_path):
    raise FileNotFoundError(f"Please ensure '{stl_path}' is in the folder.")
car_mesh = mesh.Mesh.from_file(stl_path)
orig_vectors = car_mesh.vectors.copy()

output_dir = "wormhole9_transit"
os.makedirs(output_dir, exist_ok=True)

num_frames = 121

progress_array = np.linspace(-12.0, 12.0, num_frames)


def funnel_pinch(l):
    """Transverse (y,z) compression factor along the wormhole axis at axial position l."""
    r_metric = np.sqrt(b0**2 + l**2)
    base_scale = r_metric / (np.abs(l) + b0)
    return 1.0 + (base_scale - 1.0) * warp_strength


# 2. Process frame by frame
for frame_idx, progress in enumerate(progress_array):
    processed_triangles = []
    print("progress", progress)

    for i in range(len(orig_vectors)):
        # Axial positions of the triangle's 3 vertices (relative to a mouth centre)
        v_l_vars = [progress + orig_vectors[i][j][0] for j in range(3)]

        # All vertices on one side? Otherwise it is splitting the throat.
        all_in_A = all(v <  0 for v in v_l_vars)
        all_in_B = all(v >= 0 for v in v_l_vars)
        is_splitting = not (all_in_A or all_in_B)

        # Snap the whole triangle to whichever side holds the majority,
        # so no triangle ever spans the two spheres.
        majority_in_A = sum(1 for v in v_l_vars if v < 0) >= 2
        side_is_A = all_in_A or (is_splitting and majority_in_A)

        triangle_vertices = []
        for j in range(3):
            x_local, y_local, z_local = orig_vectors[i][j]
            v_l = progress + x_local

            # If the triangle straddles the throat, clip the offending
            # vertex to the boundary so it never crosses to the other mouth.
            if is_splitting:
                if side_is_A and v_l >= 0:
                    v_l = -0.001
                elif (not side_is_A) and v_l < 0:
                    v_l =  0.0

            # ---- Metric distortion ----
            r_metric = np.sqrt(b0**2 + v_l**2)
            base_scale = r_metric / (np.abs(v_l) + b0)
            scale_factor = 1.0 + (base_scale - 1.0) * warp_strength

            # Spherical swallow: squeeze the vertex toward the mouth centre
            # when it is inside the throat sphere.
            rho = np.sqrt(y_local**2 + z_local**2)
            d = np.sqrt(v_l**2 + rho**2)
            if d < b0:
                depth = 1.0 - d / b0            # 1 at centre, 0 at surface
                tpinch = 0.1 + 0.9 * (d / b0)   # 0.1 at centre, 1.0 at surface
                scale_factor *= tpinch

            if v_l < 0:
                # Render at Wormhole 1 (disappears into it)
                pos = W1_center + np.array([v_l, y_local * scale_factor, z_local * scale_factor])
            else:
                # Render at Wormhole 2 (reappears from it)
                pos = W2_center + np.array([v_l, y_local * scale_factor, z_local * scale_factor])

            triangle_vertices.append(pos)

        processed_triangles.append(triangle_vertices)

    # 3. Export frame
    frame_vectors = np.array(processed_triangles)
    modified_mesh = mesh.Mesh(np.zeros(frame_vectors.shape[0], dtype=mesh.Mesh.dtype))
    modified_mesh.vectors = frame_vectors
    modified_mesh.save(os.path.join(output_dir, f"clean_teleport_{frame_idx:03d}.stl"))

print(f"Exported {num_frames} frames - car disappears at sphere 1 and reappears at sphere 2!")

# ============================================================
# Part 2: Spacetime warpage grid (from wormhole_grid.py)
# ============================================================

x_min, x_max = -20.0, 30.0
y_min, y_max = -10.0, 10.0
nx, ny = 100, 40

x_1d = np.linspace(x_min, x_max, nx)
y_1d = np.linspace(y_min, y_max, ny)
xx, yy = np.meshgrid(x_1d, y_1d)
zz = np.zeros_like(xx)

warped_x = xx.copy()
warped_y = yy.copy()

for center in [W1_center, W2_center]:
    dx = xx - center[0]
    dy = yy - center[1]
    r = np.sqrt(dx**2 + dy**2)
    r_safe = np.maximum(r, 0.01)

    r_metric = np.sqrt(b0**2 + r_safe**2)
    base_scale = r_metric / (r_safe + b0)
    scale = 1.0 + (base_scale - 1.0) * warp_strength
    influence = np.exp(-(r_safe / (2.0 * b0))**2)

    displacement = (scale - 1.0) * influence
    warped_x += dx * displacement
    warped_y += dy * displacement

grid = pv.StructuredGrid(
    warped_x.reshape(ny, nx),
    warped_y.reshape(ny, nx),
    zz.reshape(ny, nx),
)

grid.save("wormhole_grid.vtk")
print(f"Saved wormhole_grid.vtk  ({nx}x{ny} = {nx*ny} points)")
