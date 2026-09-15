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

output_dir = "wormhole13_transit"
os.makedirs(output_dir, exist_ok=True)

num_frames = 101
progress_array = np.linspace(-10.0, 10.0, num_frames)


def flat_point(progress, x_local, y_local, z_local, y_off):
    return np.array([progress + x_local, y_local + y_off, z_local])


def warp_ambient(p):
    """Ambient spacetime warp, exactly the grid's law: the radial factor
    per mouth (s*w inside the throat, 1+(s-1)*loc(r) outside) superposed
    over both mouths. No teleport. A point that stays outside the throat
    spheres (r > b0) is only bent by this field and passes by the wells."""
    out = p.copy()
    for center in [W1_center, W2_center]:
        dr = p - center
        r = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)
        s = funnel_pinch(r)
        if r < b0:
            w = (r / b0) ** 2
            factor = s * w
        else:
            factor = 1.0 + (s - 1.0) * loc(r)
        out += dr * (factor - 1.0)
    return out


def render_car(orig_vectors, progress, y_off, captured):
    """Render one car. Identical law for every car; the only thing that
    differs is the start position (y_off). A triangle teleports the first
    time most of its vertices physically enter W1's throat sphere
    (r < b0); from then on it is emitted from W2. Triangles that never
    enter the throat keep the pure ambient warp, so they pass by."""
    triangles = []
    for i in range(len(orig_vectors)):
        verts = []
        for j in range(3):
            x_local, y_local, z_local = orig_vectors[i][j]
            verts.append(flat_point(progress, x_local, y_local, z_local, y_off))

        if not captured[i] and sum(1 for p in verts
                                   if np.linalg.norm(p - W1_center) < b0) >= 2:
            captured[i] = True

        tri = []
        for p in verts:
            if captured[i]:
                q = W2_center + (p - W1_center)
                tri.append(warp_ambient(q))
            else:
                tri.append(warp_ambient(p))
        triangles.append(tri)
    return triangles


captured_main = [False] * len(orig_vectors)
captured_ref = [False] * len(orig_vectors)

for frame_idx, progress in enumerate(progress_array):
    print("progress", progress)

    main_triangles = render_car(orig_vectors, progress, 0.0, captured_main)
    ref_triangles = render_car(orig_vectors, progress, 4.0, captured_ref)

    all_triangles = main_triangles + ref_triangles

    frame_vectors = np.array(all_triangles)
    modified_mesh = mesh.Mesh(np.zeros(frame_vectors.shape[0], dtype=mesh.Mesh.dtype))
    modified_mesh.vectors = frame_vectors
    modified_mesh.save(os.path.join(output_dir, f"clean_teleport_{frame_idx:03d}.stl"))

print(f"Exported {num_frames} frames - both cars via shared ambient warp + throat-entry teleport")