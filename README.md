# rock-GAN
## Project Overview 

This script trains a Generalized Adversarial Network (GAN) to generate rock wall hold voxels. We then turn these voxels into meshes and save those meshes as STL files to 3D print the holds. 

The real data used to train the GAN are open-source STL files of rock wall holds from Thingiverse. 

Ultimately, it was very difficult for our model to learn from our entire dataset of 220 holds, so instead we give it a smaller amount of data, usually somewhere between 3 and 10 holds to generate new data from. This also made it easier to verify that our GAN was actually generating something new!

## README Table of Contents 

1. Installation
2. Usage Guide
3. Data pre-processing 
4. Loading the data into a custom dataloader
5. Defining the model architecture
6. Defining training parameters, weight intitialization and loss functions
7. Output post-processing
8. Output visualization
9. Output gallery
10. Citations

## Data pre-processing
Pre-processing consists of converting a directory of STL files of rock wall holds into scaled voxels using Trimesh. We then save matrix representations of these voxels using numpy arrays. These numpy arrays are stored in separate .npy files. Our custom dataloader class retrieves the data from one npy file at a time.

### Raw Mesh Data 

### Our raw mesh data is loaded using trimesh. We downloaded a series of rock wall hold STL files from Thingiverse to obtain our original mesh data. Then, we used Trimesh to load the meshes from the STLs, voxelize the meshes, and save a matrix representation from each voxel as a numpy array. 

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
                # ... mesh manipulation code ...
```


The voxels in the dataloader serve as our real data in the GAN training. To train the generator and the discriminator, we feed the generator random noise and then feed the output from the generator into the discriminator. We also feed the discriminator real data so it can make predictions about what is real and fake. The discriminator determines whether the output is real or fake, and then the generator and discriminator loss functions point them in the direction of correctness. For the generator, that means consolidating the noise into a convincing rock wall hold, and for the discriminator, that means identifying if the data it recieves is fake or real. The discriminator puts pressure on the generator to create more convincing holds so the generator can trick it. The two networks are in competition with each other. By the end of training, when we do a forward pass of the generator and input random noise, the output is a convincing rock wall hold! Whether these holds are printable varies based on the quality and continuity of the hold. Ideally, the generator learns to produce continuous holds because the training data is continuous.
In post processing, we take the output of the generator, turn it into a trimesh voxel, and then use the marching cubes algorithm to convert it into a mesh. This mesh is then saved in an STL file that we can use to 3D print the generated rock wall holds.

Now, let's get into the script!
