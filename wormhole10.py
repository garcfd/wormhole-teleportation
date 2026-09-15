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
# Part 1: Car transit animation (spherical-swallow version)
# The throat is treated as a true sphere: a vertex captured inside
# the sphere collapses radially toward the mouth centre, so the car
# disappears into a spherical surface and never pancakes into a disc.
# ============================================================

# 1. Load the car mesh
stl_path = 'car4.stl'
if not os.path.exists(stl_path):
    raise FileNotFoundError(f"Please ensure '{stl_path}' is in the folder.")
car_mesh = mesh.Mesh.from_file(stl_path)
orig_vectors = car_mesh.vectors.copy()

output_dir = "wormhole10_transit"
os.makedirs(output_dir, exist_ok=True)

num_frames = 101

progress_array = np.linspace(-10.0, 10.0, num_frames)


def funnel_pinch(l):
    """Transverse (y,z) compression factor along the wormhole axis at axial position l."""
    r_metric = np.sqrt(b0**2 + l**2)
    base_scale = r_metric / (np.abs(l) + b0)
    return 1.0 + (base_scale - 1.0) * warp_strength


def render_vertex(center, l, y, z):
    """Render one vertex relative to a mouth centre.
    The funnel pinch applies everywhere; additionally, inside the throat
    sphere (d < b0) the whole position collapses radially toward the centre.
    The collapse factor w is 1 at the sphere surface (== pinch only) and
    0 at the centre, so there is no scale discontinuity at the boundary."""
    rho = np.sqrt(y * y + z * z)
    d = np.sqrt(l * l + rho * rho)

    s = funnel_pinch(l)
    px, py, pz = l, y * s, z * s

    if d < b0:
        w = (d / b0) ** 2   # 1 at the sphere surface, 0 at its centre
        px, py, pz = px * w, py * w, pz * w

    return center + np.array([px, py, pz])


# 2. Process frame by frame
for frame_idx, progress in enumerate(progress_array):
    processed_triangles = []
    print("progress", progress)

    for i in range(len(orig_vectors)):
        # Axial positions of the triangle's 3 vertices (relative to a mouth centre)
        l_vars = [progress + orig_vectors[i][j][0] for j in range(3)]

        # Snap the whole triangle to one mouth so it never spans the two spheres.
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

    # 3. Export frame
    frame_vectors = np.array(processed_triangles)
    modified_mesh = mesh.Mesh(np.zeros(frame_vectors.shape[0], dtype=mesh.Mesh.dtype))
    modified_mesh.vectors = frame_vectors
    modified_mesh.save(os.path.join(output_dir, f"clean_teleport_{frame_idx:03d}.stl"))

print(f"Exported {num_frames} frames - spherical swallow at sphere 1, emergence at sphere 2!")

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

# Use the SAME transform as render_vertex() above (funnel_pinch applied to the
# transverse coordinate as a function of axial distance l, plus the spherical
# collapse near the throat), evaluated on the z=0 slice. This is what makes
# the grid geometrically consistent with the STL transit warp: it only
# compresses the transverse (here: y) direction as a function of axial (x)
# distance from a mouth center, rather than an isotropic radial scaling of
# both x and y together (which is a different, inconsistent geometry).
#
# Each grid point is warped relative to whichever mouth it's nearer to along
# x (split at the midpoint between the two mouths), mirroring the way the
# STL code snaps each triangle to a single mouth's frame. By symmetry, the
# two halves agree exactly at the seam, so there's no visible discontinuity.
midpoint_x = (W1_center[0] + W2_center[0]) / 2.0

warped_x = np.zeros_like(xx)
warped_y = np.zeros_like(yy)

for center, mask in [
    (W1_center, xx < midpoint_x),
    (W2_center, xx >= midpoint_x),
]:
    l = xx - center[0]   # axial offset from this mouth (matches render_vertex's l)
    y = yy - center[1]   # transverse offset (z implicitly 0 on this slice)

    rho = np.abs(y)
    d = np.sqrt(l**2 + rho**2)

    s = funnel_pinch(l)
    px, py = l.copy(), y * s

    inside = d < b0
    w = (d / b0) ** 2
    px = np.where(inside, px * w, px)
    py = np.where(inside, py * w, py)

    warped_x = np.where(mask, center[0] + px, warped_x)
    warped_y = np.where(mask, center[1] + py, warped_y)

grid = pv.StructuredGrid(
    warped_x.reshape(ny, nx),
    warped_y.reshape(ny, nx),
    zz.reshape(ny, nx),
)

grid.save("wormhole_grid.vtk")
print(f"Saved wormhole_grid.vtk  ({nx}x{ny} = {nx*ny} points)")
