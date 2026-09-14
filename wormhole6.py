import numpy as np
from stl import mesh
import os

# 1. Load the car mesh
stl_path = 'car.stl'
if not os.path.exists(stl_path):
    raise FileNotFoundError(f"Please ensure '{stl_path}' is in the folder.")
car_mesh = mesh.Mesh.from_file(stl_path)
orig_vectors = car_mesh.vectors.copy()

b0 = 1.5  # Radius of both wormhole mouths
W1_center = np.array([ 0.0, 0.0, 0.0])  # Entrance Mouth
W2_center = np.array([10.0, 0.0, 0.0])  # Exit Mouth

output_dir = "wormhole_perfect_transit"
os.makedirs(output_dir, exist_ok=True)

num_frames = 100

progress_array = np.linspace(-10.0, 10.0, num_frames)

# 2. Process frame by frame
for frame_idx, progress in enumerate(progress_array):
    processed_triangles = []
    
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
            scale_factor = r_metric / (np.abs(v_l) + b0)
            
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

