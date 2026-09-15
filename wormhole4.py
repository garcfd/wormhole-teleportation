import numpy as np
from stl import mesh
import os

# 1. Load the car mesh
stl_path = 'car.stl'
if not os.path.exists(stl_path):
    raise FileNotFoundError(f"Please ensure '{stl_path}' is in the folder.")
car_mesh = mesh.Mesh.from_file(stl_path)
orig_vectors = car_mesh.vectors.copy()

# 2. Setup the Two Wormhole Spheres in space
b0 = 1.5  # Radius of both wormhole mouths
W1_center = np.array([0.0, 0.0, 0.0])   # Entrance Mouth
W2_center = np.array([12.0, 0.0, 0.0])  # Exit Mouth (12 units away)

num_frames = 90
output_dir = "wormhole_teleportation"
os.makedirs(output_dir, exist_ok=True)

# 3. Animate the transit timeline
for frame_idx in range(num_frames):
    frame_vectors = orig_vectors.copy()
    
    # Progress goes from -8 (far before W1) to +8 (far after W2)
    progress = np.linspace(-8.0, 8.0, num_frames)[frame_idx]
    
    for i in range(len(frame_vectors)):
        for j in range(3):
            # Original local vertex of the car mesh
            x_local, y_local, z_local = frame_vectors[i][j]
            
            # Position of this specific vertex along the continuous wormhole throat axis
            v_l = progress + x_local
            
            if v_l < 0:
                # --- PHASE 1: Object is entering Wormhole 1 ---
                # Calculate distortion relative to the entrance throat
                r_metric = np.sqrt(b0**2 + v_l**2)
                scale_factor = r_metric / (np.abs(v_l) + b0)
                
                # Model the car driving along -X towards W1 center
                v_global = W1_center + np.array([v_l, y_local * scale_factor, z_local * scale_factor])
                
            else:
                # --- PHASE 2 & 3: Object topology seamlessly teleports to Wormhole 2 ---
                # Calculate distortion relative to the exit throat
                r_metric = np.sqrt(b0**2 + v_l**2)
                scale_factor = r_metric / (np.abs(v_l) + b0)
                
                # Model the car emerging along +X away from W2 center
                v_global = W2_center + np.array([v_l, y_local * scale_factor, z_local * scale_factor])
            
            frame_vectors[i][j] = v_global

    # Save frame safely using the integer polygon shape count
    modified_mesh = mesh.Mesh(np.zeros(frame_vectors.shape[0], dtype=mesh.Mesh.dtype))
    modified_mesh.vectors = frame_vectors
    
    output_mesh_path = os.path.join(output_dir, f"teleport_frame_{frame_idx:03d}.stl")
    modified_mesh.save(output_mesh_path)

print(f"Successfully generated {num_frames} frames with seamless wormhole teleportation!")

