import numpy as np
from stl import mesh
import os

# 1. Helper function to generate a sphere mesh
def create_sphere(center, radius, subdivisions=16):
    u = np.linspace(0, 2 * np.pi, subdivisions)
    v = np.linspace(0, np.pi, subdivisions)
    
    # Generate vertices
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Construct triangles
    triangles = []
    for i in range(subdivisions - 1):
        for j in range(subdivisions - 1):
            p1 = [x[i, j], y[i, j], z[i, j]]
            p2 = [x[i+1, j], y[i+1, j], z[i+1, j]]
            p3 = [x[i, j+1], y[i, j+1], z[i, j+1]]
            p4 = [x[i+1, j+1], y[i+1, j+1], z[i+1, j+1]]
            triangles.append([p1, p2, p3])
            triangles.append([p2, p4, p3])
            
    sphere_mesh = mesh.Mesh(np.zeros(len(triangles), dtype=mesh.Mesh.dtype))
    sphere_mesh.vectors = np.array(triangles)
    return sphere_mesh

# 2. Main setup and configuration
stl_path = 'car.stl'
if not os.path.exists(stl_path):
    raise FileNotFoundError(f"Please ensure '{stl_path}' is in the folder.")
car_mesh = mesh.Mesh.from_file(stl_path)
orig_vectors = car_mesh.vectors.copy()

b0 = 1.5  # Radius of both wormhole mouths
W1_center = np.array([0.0, 0.0, 0.0])   # Entrance Mouth
W2_center = np.array([12.0, 0.0, 0.0])  # Exit Mouth

output_dir = "wormhole_with_spheres"
os.makedirs(output_dir, exist_ok=True)

# Export the static wormhole geometry files
# create_sphere(W1_center, b0).save(os.path.join(output_dir, "wormhole_1.stl"))
# create_sphere(W2_center, b0).save(os.path.join(output_dir, "wormhole_2.stl"))

num_frames = 90
# 3. Animate the transit timeline
for frame_idx in range(num_frames):
    frame_vectors = orig_vectors.copy()
    progress = np.linspace(-8.0, 8.0, num_frames)[frame_idx]
    
    for i in range(len(frame_vectors)):
        for j in range(3):
            x_local, y_local, z_local = frame_vectors[i][j]
            v_l = progress + x_local
            
            # Metric distortion factor
            r_metric = np.sqrt(b0**2 + v_l**2)
            scale_factor = r_metric / (np.abs(v_l) + b0)
            
            if v_l < 0:
                # Entering Wormhole 1
                v_global = W1_center + np.array([v_l, y_local * scale_factor, z_local * scale_factor])
            else:
                # Exiting Wormhole 2
                v_global = W2_center + np.array([v_l, y_local * scale_factor, z_local * scale_factor])
            
            frame_vectors[i][j] = v_global

    modified_mesh = mesh.Mesh(np.zeros(frame_vectors.shape[0], dtype=mesh.Mesh.dtype))
    modified_mesh.vectors = frame_vectors
    modified_mesh.save(os.path.join(output_dir, f"teleport_frame_{frame_idx:03d}.stl"))

print(f"Exported car sequence and static spheres to '{output_dir}/'!")

