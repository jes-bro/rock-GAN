# rock-GAN
## Project Overview 

This script trains a Generalized Adversarial Network (GAN) to generate rock wall hold voxels. We then turn these voxels into meshes and save those meshes as STL files to 3D print the holds. 

The real data used to train the GAN are open-source STL files of rock wall holds from Thingiverse. 

Ultimately, it was very difficult for our model to learn from our entire dataset of 220 holds, so instead we give it a smaller amount of data, usually somewhere between 3 and 10 holds to generate new data from. This also made it easier to verify that our GAN was actually generating something new!

## README Table of Contents 

1. Installation
2. Usage Guide
3. Data pre-processing         
4. Loading the data using a custom dataloader
5. Defining the model architecture
6. Defining training parameters, weight intitialization and loss functions
7. Training Loop
8. Post-processing
9. Visualization
10. Output gallery
11. Bloopers
12. Citations

## Data pre-processing
Pre-processing consists of converting a directory of STL files of rock wall holds into scaled voxels using Trimesh. We then save matrix representations of these voxels using numpy arrays. These numpy arrays are stored in separate .npy files. Our custom dataloader class retrieves the data from one npy file at a time.

### Raw Mesh Data 

Our raw mesh data is loaded using trimesh. We downloaded a series of rock wall hold STL files from Thingiverse to obtain our original mesh data. Then, we used Trimesh to load the meshes from the STLs, voxelize the meshes, and save a matrix representation from each voxel as a numpy array. 

Our pre-processing code is as follows: 
```python
file_name_counter = 0

folderpath = 'unprocessed_data'
os.makedirs('processed_npys', exist_ok=True)
for filename in os.listdir(folderpath):
        file_path = os.path.join(folderpath, filename)
        if os.path.isfile(file_path):  # Check if it's a file
                mesh = trimesh.load(file_path)
                # .
                # .
                # ... Normalizing, scaling, storing code ...
```
### Normalizing and Scaling

We load each mesh and normalize the vertices to be within a sphere with a radius of one, centered at (0, 0, 0). Then, we voxelize the mesh and scale it to be within a range of 32x32x32. We used the following [pytorch3D tutorial](https://colab.research.google.com/github/facebookresearch/pytorch3d/blob/stable/docs/tutorials/deform_source_mesh_to_target_mesh.ipynb) as the initial inspiration for how to do this, but ended up using Trimesh instead. 

Our normalizing and scaling code is as follows: 
```python
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
```

### Saving voxelized data
For each real rock wall hold, we created an empty 32x32x32 matrix and then populated that matrix with the scaled voxel. We did this by padding the voxel to match the dimensions of the matrix and then setting the matrix equal to the voxel. 
Here is how we did it:
```python
# Convert the voxel into a np array and pad it
voxel_matrix = voxel.matrix

# Cropping the matrix to make sure it's within 32x32x32
voxel_matrix = voxel_matrix[0:32, 0:32, 0:32]

reshaped_voxel = np.zeros((32, 32, 32))

if any(dim > 32 for dim in voxel_matrix.shape):
        print("ERROR: Voxel Exceeded 32x32x32. Skipping")
        continue

# Find the required voxel padding
x_padding = int(math.floor((32 - voxel_matrix.shape[0]) / 2))
y_padding = int(math.floor((32 - voxel_matrix.shape[1]) / 2))
z_padding = int(math.floor((32 - voxel_matrix.shape[2]) / 2))

# Put the voxel occupancy in its designated spot in the empty 3D Matrix
reshaped_voxel[x_padding: x_padding + voxel_matrix.shape[0], y_padding: y_padding + voxel_matrix.shape[1], z_padding: z_padding + voxel_matrix.shape[2]] = voxel_matrix
```
We save each voxel's matrix representation to a separate numpy file. Each npy file saves a single real rock wall hold voxel. Each numpy file is named with its corrresponding index in the data. 

Here is how we save the data:
```python
# Save reshaped voxel as a numpy array
np.save(f'processed_npys/{file_name_counter}.npy', reshaped_voxel)
```

### Loading the data into a custom dataloader

To load this data, we wrote a custom dataloader. In the get_item function, we retrieve each data point using its index to load it from the file it is stored in.

Here is how we load it in the get_item function:
```python
def __getitem__(self, idx):
        # print(self.data[idx].shape)
        idx = idx +
        data_pt = np.load(f"/content/drive/My Drive/TanGen Data/processed_npys/{idx}.npy")
        print(f"/content/drive/My Drive/TanGen Data/processed_npys/{idx}.npy")
        dp = torch.tensor(data_pt, dtype=torch.float32)
        return dp.unsqueeze(0)
```

### Defining the model architecture

### Training Loop
To train the generator and the discriminator, we feed the generator random noise. Then, we feed the output from the generator into the discriminator. We also feed the discriminator real data. The discriminator determines whether output is real or fake, and then the generator and discriminator loss functions point them in the direction of correctness. For the generator, that means consolidating the noise into a convincing rock wall hold, and for the discriminator, that means identifying if the data it recieves is fake or real. The discriminator puts pressure on the generator to create more convincing holds so the generator can trick it. The two networks are in competition with each other. 

By the end of training, when we feed the generator random noise, the output is a convincing rock wall hold! Whether these holds are printable varies based on the quality and continuity of the hold. Ideally, the generator learns to produce continuous holds because the training data is continuous.

### Post-processing
In post processing, we take the output of the generator, turn it into a trimesh voxel, and then use the marching cubes algorithm to convert it into a mesh. This mesh is then saved in an STL file that we can use to 3D print the ge`nerated rock wall holds.

Here is the code we use to do that:
```python
fake = netG(noise).squeeze()
threshold = 0.7  # We adjust this depending on whether or not it helps make the holds easier to print
binary_fake = (fake > threshold).cpu().numpy().astype(bool)
print("Voxel grid unique values:", np.unique(binary_fake))
print(fake.shape)


new_voxel = trimesh.voxel.base.VoxelGrid(binary_fake)

print(f"voxel: {new_voxel.shape}")
mc = new_voxel.marching_cubes

# Save the remeshed surface as an STL file
mc.export("test5.stl")
```

### Visualization 
We visualize the output using matplotlib. 

Here are some sample outputs!

### Bloopers
