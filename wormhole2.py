import numpy as np
from stl import mesh
import os

# 1. Load your STL File 
# Replace 'car.stl' with the path to your simplified car mesh
stl_path = 'car.stl'
if not os.path.exists(stl_path):
    raise FileNotFoundError(f"Please place a '{stl_path}' file in the directory.")

car_mesh = mesh.Mesh.from_file("car.stl")

# 2. Wormhole & Trajectory Parameters
b0 = 2.0        # Throat radius (scaled to fit the car size)
l_start = -15.0 # Start far away in Universe A
l_end = 15.0    # End in Universe B
num_frames = 60 # Number of animation frames to export

# Linearly step the car's center position through proper radial distance (l)
car_positions_l = np.linspace(l_start, l_end, num_frames)

# 3. Process Frame by Frame
# Create an output directory for the warped frames
output_dir = "wormhole_frames"
os.makedirs(output_dir, exist_ok=True)

# Flatten original vertices to transform them
# car_mesh.vectors is an array of shape (N, 3, 3) representing N triangles
orig_vectors = car_mesh.vectors.copy()

for frame_idx, center_l in enumerate(car_positions_l):
    # Create a fresh copy of the mesh structure for this frame
    frame_vectors = orig_vectors.copy()
    
    for i in range(len(frame_vectors)): # Loop through each triangle
        for j in range(3): # Loop through each vertex of the triangle
            x_orig, y_orig, z_orig = frame_vectors[i][j]
            
            # The local vertex position relative to the car's center of mass
            # x_orig represents the front-to-back length of the car
            v_l = center_l + x_orig 
            
            # Calculate the circumferential radius 'r' at this vertex's specific l-coordinate
            r_metric = np.sqrt(b0**2 + v_l**2)
            
            # Gravitational squeeze/stretch factor: 
            # Far away (large |l|), distortion goes to 1.0 (normal space).
            # At the throat (l=0), space is tightly compressed around the throat radius.
            scale_factor = r_metric / (np.abs(v_l) + b0)
            
            # Apply distortion:
            # 1. Longitudinal position shifts along the continuous wormhole axis
            frame_vectors[i][j][0] = v_l 
            
            # 2. Cross-sectional dimensions (Y and Z) are warped by the metric scaling
            frame_vectors[i][j][1] = y_orig * scale_factor
            frame_vectors[i][j][2] = z_orig * scale_factor

    # Save the distorted mesh as a new STL file
    modified_mesh = mesh.Mesh(np.zeros(frame_vectors.shape[0], dtype=mesh.Mesh.dtype))
    modified_mesh.vectors = frame_vectors
    
    output_mesh_path = os.path.join(output_dir, f"car_frame_{frame_idx:03d}.stl")
    modified_mesh.save(output_mesh_path)

print(f"Successfully exported {num_frames} frames to the '{output_dir}/' directory!")

