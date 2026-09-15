import numpy as np
from stl import mesh
import os

# 1. Load the car mesh
stl_path = 'car.stl'
if not os.path.exists(stl_path):
    raise FileNotFoundError(f"Please ensure '{stl_path}' is in the folder.")
car_mesh = mesh.Mesh.from_file(stl_path)
orig_vectors = car_mesh.vectors.copy()

# 2. Define the Multi-Wormhole System
b0 = 1.5  # Radius of both wormhole mouths

# Wormhole 1 (The Source / Exit) located at the origin
W1_center = np.array([0.0, 0.0, 0.0])

# Wormhole 2 (The Destination / Entrance) located 12 units away on X, slightly offset on Y
W2_center = np.array([12.0, 2.0, 0.0])

# 3. Define the Global Path of the Car
num_frames = 80
# The car travels along the X axis from X=-5 (inside W1) to X=17 (inside W2)
path_x = np.linspace(-5.0, 17.0, num_frames)

output_dir = "two_wormholes_transit"
os.makedirs(output_dir, exist_ok=True)

# 4. Process Frame by Frame
for frame_idx, car_x in enumerate(path_x):
    frame_vectors = orig_vectors.copy()
    
    # Calculate car's current Y coordinate based on a smooth line between W1 and W2
    # This simulates a car steering slightly toward the second mouth
    if car_x < 0:
        car_y = 0.0
    elif car_x > 12.0:
        car_y = 2.0
    else:
        # Linear interpolation between the two mouths
        car_y = (car_x / 12.0) * 2.0
        
    car_center = np.array([car_x, car_y, 0.0])
    
    for i in range(len(frame_vectors)):
        for j in range(3):
            # Global baseline coordinate of the vertex in normal space
            v_global = car_center + frame_vectors[i][j]
            
            # Check distance to Wormhole Mouth 1
            dist_to_W1 = np.linalg.norm(v_global - W1_center)
            # Check distance to Wormhole Mouth 2
            dist_to_W2 = np.linalg.norm(v_global - W2_center)
            
            # Case A: Vertex is interacting with Wormhole 1 (Emerging)
            if car_x <= 4.0 and dist_to_W1 < (b0 + 4.0):
                # Calculate metric distortion based on distance from W1 center
                l_val = dist_to_W1 - b0
                r_metric = np.sqrt(b0**2 + l_val**2)
                scale_factor = r_metric / (np.abs(l_val) + b0)
                
                # Apply radial distortion outward from the sphere center
                direction = (v_global - W1_center) / (dist_to_W1 + 1e-5)
                v_global = W1_center + direction * dist_to_W1 * scale_factor
                
            # Case B: Vertex is interacting with Wormhole 2 (Entering)
            elif car_x > 4.0 and dist_to_W2 < (b0 + 4.0):
                # Calculate metric distortion based on distance from W2 center
                l_val = dist_to_W2 - b0
                r_metric = np.sqrt(b0**2 + l_val**2)
                scale_factor = r_metric / (np.abs(l_val) + b0)
                
                # Apply radial distortion inward/outward relative to W2
                direction = (v_global - W2_center) / (dist_to_W2 + 1e-5)
                v_global = W2_center + direction * dist_to_W2 * scale_factor
            
            # Case C: Car is in the interstitial flat space (No change to v_global)
            frame_vectors[i][j] = v_global

    # Save frame
    modified_mesh = mesh.Mesh(np.zeros(frame_vectors.shape[0], dtype=mesh.Mesh.dtype))
    modified_mesh.vectors = frame_vectors
    output_mesh_path = os.path.join(output_dir, f"transit_frame_{frame_idx:03d}.stl")
    modified_mesh.save(output_mesh_path)

print(f"Generated {num_frames} frames tracking transit between separate mouths!")

