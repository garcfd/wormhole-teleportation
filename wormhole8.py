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
# Part 1: Car transit animation (spherical throat version)
# ============================================================

# 1. Load the car mesh
stl_path = 'car3.stl'
if not os.path.exists(stl_path):
    raise FileNotFoundError(f"Please ensure '{stl_path}' is in the folder.")
car_mesh = mesh.Mesh.from_file(stl_path)
orig_vectors = car_mesh.vectors.copy()

output_dir = "wormhole8_transit"
os.makedirs(output_dir, exist_ok=True)

num_frames = 101

progress_array = np.linspace(-10.0, 10.0, num_frames)


def funnel_pinch(l):
    """Transverse (y,z) compression factor along the wormhole axis at axial position l."""
    r_metric = np.sqrt(b0**2 + l**2)
    base_scale = r_metric / (np.abs(l) + b0)
    return 1.0 + (base_scale - 1.0) * warp_strength


def in_throat_scale(ax, d):
    """Transverse squeeze factor for a vertex inside the throat sphere.
    ax is its axial coordinate (mapped onto the funnel), d its 3D distance
    from the throat center.  Squeezes to ~0.1 at the sphere centre."""
    pinch = funnel_pinch(ax)
    depth = 1.0 - d / b0          # 0 at sphere surface, 1 at centre
    tpinch = 0.1 + 0.9 * (d / b0) # 1.0 at surface, 0.1 at centre
    return pinch * tpinch


# 2. Process frame by frame
for frame_idx, progress in enumerate(progress_array):
    processed_triangles = []
    print("progress", progress)

    for i in range(len(orig_vectors)):
        triangle_vertices = []
        for j in range(3):
            x_local, y_local, z_local = orig_vectors[i][j]

            # Axial position along the wormhole axis (relative to a mouth centre)
            a = progress + x_local
            rho = np.sqrt(y_local**2 + z_local**2)
            d = np.sqrt(a**2 + rho**2)   # 3D distance from the throat centre

            if d < b0:
                # ---- Inside the throat sphere: spherical swallow effect ----
                # Project onto the sphere surface along its radial direction.
                f = b0 / d
                pa = a * f               # axial coordinate on the sphere surface
                depth = 1.0 - d / b0     # 0 at surface, 1 at centre
                ax = pa * (1.0 - depth)  # axial coordinate inside the funnel

                scale = in_throat_scale(ax, d)

                if a < 0:
                    pos = W1_center + np.array([ax, y_local * scale, z_local * scale])
                else:
                    pos = W2_center + np.array([ax, y_local * scale, z_local * scale])
            else:
                # ---- Outside the throat sphere ----
                scale = funnel_pinch(a)

                if a < 0:
                    # Render in Wormhole 1
                    pos = W1_center + np.array([a, y_local * scale, z_local * scale])
                else:
                    # Render in Wormhole 2
                    pos = W2_center + np.array([a, y_local * scale, z_local * scale])

            triangle_vertices.append(pos)

        processed_triangles.append(triangle_vertices)

    # 3. Export frame
    frame_vectors = np.array(processed_triangles)
    modified_mesh = mesh.Mesh(np.zeros(frame_vectors.shape[0], dtype=mesh.Mesh.dtype))
    modified_mesh.vectors = frame_vectors
    modified_mesh.save(os.path.join(output_dir, f"clean_teleport_{frame_idx:03d}.stl"))

print(f"Exported {num_frames} frames with the spherical-throat transit!")

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