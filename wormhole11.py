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


def funnel_pinch(l):
    """Transverse (y,z) compression factor along the wormhole axis at axial position l."""
    r_metric = np.sqrt(b0**2 + l**2)
    base_scale = r_metric / (np.abs(l) + b0)
    return 1.0 + (base_scale - 1.0) * warp_strength


# ============================================================
# Part 1: Spacetime warpage grid
# ============================================================

x_min, x_max = -20.0, 30.0
y_min, y_max = -10.0, 10.0
nx, ny = 100, 40

x_1d = np.linspace(x_min, x_max, nx)
y_1d = np.linspace(y_min, y_max, ny)
xx, yy = np.meshgrid(x_1d, y_1d)
zz = np.zeros_like(xx)


def loc(r):
    return np.exp(-((r - b0) / (1.5 * b0)) ** 2)


warped_x = xx.copy()
warped_y = yy.copy()

for center in [W1_center, W2_center]:
    dx = xx - center[0]
    dy = yy - center[1]
    r = np.sqrt(dx**2 + dy**2)

    s = funnel_pinch(r)
    w = (r / b0) ** 2

    inside = r < b0
    factor_inside = s * w
    factor_outside = 1.0 + (s - 1.0) * loc(r)
    factor = np.where(inside, factor_inside, factor_outside)

    warped_x += dx * (factor - 1.0)
    warped_y += dy * (factor - 1.0)

grid = pv.StructuredGrid(
    warped_x.reshape(ny, nx),
    warped_y.reshape(ny, nx),
    zz.reshape(ny, nx),
)

grid.save("wormhole_grid.vtk")
print(f"Saved wormhole_grid.vtk  ({nx}x{ny} = {nx*ny} points)")

# ============================================================
# Part 2: Car transit animation
# ============================================================

stl_path = 'car4.stl'
if not os.path.exists(stl_path):
    raise FileNotFoundError(f"Please ensure '{stl_path}' is in the folder.")
car_mesh = mesh.Mesh.from_file(stl_path)
orig_vectors = car_mesh.vectors.copy()

output_dir = "wormhole10_transit"
os.makedirs(output_dir, exist_ok=True)

num_frames = 101
progress_array = np.linspace(-10.0, 10.0, num_frames)


def render_vertex(center, l, y, z):
    """Render one vertex relative to a mouth centre.
    The funnel pinch applies everywhere; inside the throat sphere
    (d < b0) the whole position collapses radially toward the centre
    scaling to a point at d=0."""
    rho = np.sqrt(y * y + z * z)
    d = np.sqrt(l * l + rho * rho)

    s = funnel_pinch(l)
    px, py, pz = l, y * s, z * s

    if d < b0:
        w = (d / b0) ** 2
        px, py, pz = px * w, py * w, pz * w

    return center + np.array([px, py, pz])


for frame_idx, progress in enumerate(progress_array):
    processed_triangles = []
    print("progress", progress)

    for i in range(len(orig_vectors)):
        l_vars = [progress + orig_vectors[i][j][0] for j in range(3)]

        all_in_A = all(l <  0 for l in l_vars)
        all_in_B = all(l >= 0 for l in l_vars)
        is_splitting = not (all_in_A or all_in_B)
        majority_in_A = sum(1 for l in l_vars if l < 0) >= 2
        side_is_A = all_in_A or (is_splitting and majority_in_A)

        triangle_vertices = []
        for j in range(3):
            x_local, y_local, z_local = orig_vectors[i][j]
            l = progress + x_local

            if side_is_A:
                pos = render_vertex(W1_center, l, y_local, z_local)
            else:
                pos = render_vertex(W2_center, l, y_local, z_local)

            triangle_vertices.append(pos)

        processed_triangles.append(triangle_vertices)

    frame_vectors = np.array(processed_triangles)
    modified_mesh = mesh.Mesh(np.zeros(frame_vectors.shape[0], dtype=mesh.Mesh.dtype))
    modified_mesh.vectors = frame_vectors
    modified_mesh.save(os.path.join(output_dir, f"clean_teleport_{frame_idx:03d}.stl"))

print(f"Exported {num_frames} frames - spherical swallow at sphere 1, emergence at sphere 2!")
