"""
Script for pre-processing meshes prior to use in the TanGen GAN network
training. 

This script takes in meshes and:
1. Makes a folder called "processed_npys"
2. Centers the meshes
3. Normalizes them within a unit sphere
4. Converts them to 32x32x32 binary voxel arrays through cropping and padding
5. Saves the voxel arrays as numpy files
6. Repeats this for all meshes in the specified source folder 
        ("unprocessed_data" by default)
"""

import trimesh
import numpy as np
import os
import math
file_name_counter = 0

# Specify the path to the folder with your unprocessed meshes below
folderpath = 'unprocessed_data'
os.makedirs('processed_npys', exist_ok=True)
for filename in os.listdir(folderpath):
        file_path = os.path.join(folderpath, filename)
        if os.path.isfile(file_path):  # Check if it's a file
                mesh = trimesh.load(file_path)

                # Extract the faces and vertices
                verts = mesh.vertices   # (V, 3) array of vertices
                faces_idx = mesh.faces  # (F, 3) array of face indices

                # # Center and normalize the mesh
                center = mesh.center_mass
                verts_centered = verts - center

                scale = np.abs(verts_centered).max()
                verts_normalized = verts/scale

                # Remesh the normalized vertices
                normalized_mesh = trimesh.Trimesh(vertices=verts_normalized, faces=faces_idx)

                # Voxelize the mesh
                voxel = trimesh.voxel.creation.voxelize(normalized_mesh, 0.0625, method="subdivide")
                
                # Convert the voxel into a np array
                voxel_matrix = voxel.matrix

                # Cropping the matrix to make sure it's within 32x32x32
                voxel_matrix = voxel_matrix[0:32, 0:32, 0:32]

                reshaped_voxel = np.zeros((32, 32, 32))

                if any(dim > 32 for dim in voxel_matrix.shape):
                        print("ERROR: Voxel Exceeded 32x32x32. Skipping")
                        continue

                # Pad the voxel array to make sure it is 32x32x32
                # Calculate the required padding on all sides of the model
                x_padding = int(math.floor((32 - voxel_matrix.shape[0]) / 2))
                y_padding = int(math.floor((32 - voxel_matrix.shape[1]) / 2))
                z_padding = int(math.floor((32 - voxel_matrix.shape[2]) / 2))

                # Put the voxel occupancy in its designated spot in the empty 3D Matrix
                reshaped_voxel[x_padding: x_padding + voxel_matrix.shape[0], y_padding: y_padding + voxel_matrix.shape[1], z_padding: z_padding + voxel_matrix.shape[2]] = voxel_matrix

                # Save reshaped voxel as a numpy array
                np.save(f'processed_npys/{file_name_counter}.npy', reshaped_voxel) 

                print(f"Processed: {filename} as {file_name_counter}.npy")
                # Increase the file name number
                file_name_counter += 1

print("All Done!")