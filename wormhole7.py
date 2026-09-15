import numpy as np
from stl import mesh
import os
import pyvista as pv

# ============================================================
# Shared wormhole parameters
# ============================================================
b0 = 1.5  # Radius of both wormhole mouths
warp_strength = 3.0  # Amplification of the warpage effect (1.0 = original)
W1_center = np.array([ 0.0, 0.0, 0.0])  # Entrance Mouth
W2_center = np.array([10.0, 0.0, 0.0])  # Exit Mouth

# ============================================================
# Part 1: Car transit animation (from wormhole6.py)
# ============================================================

# 1. Load the car mesh
stl_path = 'car3.stl'
if not os.path.exists(stl_path):
    raise FileNotFoundError(f"Please ensure '{stl_path}' is in the folder.")
car_mesh = mesh.Mesh.from_file(stl_path)
orig_vectors = car_mesh.vectors.copy()

output_dir = "wormhole_perfect_transit"
os.makedirs(output_dir, exist_ok=True)

num_frames = 101

progress_array = np.linspace(-10.0, 10.0, num_frames)

# 2. Process frame by frame
for frame_idx, progress in enumerate(progress_array):
    processed_triangles = []
    print("progress", progress)

    for i in range(len(orig_vectors)):
        # Calculate the continuous metric coordinate (v_l) for all 3 vertices of this triangle
        v_l_vars = [progress + orig_vectors[i][j][0] for j in range(3)]

        # Check if the triangle is splitting across the throat boundary
        all_in_A = all(v <  0 for v in v_l_vars)
        all_in_B = all(v >= 0 for v in v_l_vars)

        # If a single triangle is stretching across both sides, we force it to align
        # with whichever side its majority rests on to prevent stretching lines.
        is_splitting = not (all_in_A or all_in_B)

        triangle_vertices = []
        for j in range(3):
            x_local, y_local, z_local = orig_vectors[i][j]
            v_l = progress + x_local

            # If splitting, clip the offending vertex to the throat boundary to avoid stretching
            if is_splitting:
                # If the majority of the triangle is in A, snap the B vertices to 0, and vice versa
                majority_in_A = sum(1 for v in v_l_vars if v < 0) >= 2
                if majority_in_A and v_l >= 0:
                    v_l = -0.001
                elif (not majority_in_A) and v_l < 0:
                    v_l =  0.0

            # Metric distortion tracking
            r_metric = np.sqrt(b0**2 + v_l**2)
            base_scale = r_metric / (np.abs(v_l) + b0)
            scale_factor = 1.0 + (base_scale - 1.0) * warp_strength

            if v_l < 0:
                # Render perfectly at Wormhole 1
                v_global = W1_center + np.array([v_l, y_local * scale_factor, z_local * scale_factor])
            else:
                # Render perfectly at Wormhole 2
                v_global = W2_center + np.array([v_l, y_local * scale_factor, z_local * scale_factor])

            triangle_vertices.append(v_global)

        processed_triangles.append(triangle_vertices)

    # 3. Export frame
    frame_vectors = np.array(processed_triangles)
    modified_mesh = mesh.Mesh(np.zeros(frame_vectors.shape[0], dtype=mesh.Mesh.dtype))
    modified_mesh.vectors = frame_vectors
    modified_mesh.save(os.path.join(output_dir, f"clean_teleport_{frame_idx:03d}.stl"))

print(f"Exported {num_frames} clean frames with zero geometric stretching!")

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

    influence = 1.0 / (1.0 + (r_safe / (3.0 * b0))**2)

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