# This script processes a TensorFlow Dataset (TFDS) named "symmetric_solids," designed 
# for 3D pose estimation tasks. The dataset consists of symmetric 3D shapes, each rendered 
# with different orientations. It includes both featureless and marked shapes with varying 
# degrees of symmetry. Below is a detailed overview of the dataset's structure and features:

# Dataset Overview:
# - Name: symmetric_solids
# - Description: 3D pose estimation dataset with symmetric shapes, including both featureless 
#   and marked shapes. The dataset provides equivalent rotations for featureless shapes for 
#   evaluation.
# - Total Dataset Size: ~3.94 GiB
# - Features:
#   - 'image': A 224x224 RGB image of the 3D shape.
#   - 'label_shape': The shape type (8 classes).
#   - 'rotation': The rotation matrix (3x3) used for rendering.
#   - 'rotations_equivalent': Known equivalent rotations for evaluation (if available).
# - Splits:
#   - Train: 360,000 examples
#   - Test: 40,000 examples
# - Citation: Implicit Representation of Probability Distributions on the Rotation Manifold 
#   (ICML 2021)

# Note: This dataset is useful for benchmarking 3D pose estimation models, particularly for 
# handling symmetric objects with visually indistinguishable orientations.

# Dataset Info: tfds.core.DatasetInfo(
#     name='symmetric_solids',
#     full_name='symmetric_solids/1.0.0',
#     description="""
#     This is a pose estimation dataset, consisting of symmetric 3D shapes where
#     multiple orientations are visually indistinguishable. The challenge is to
#     predict all equivalent orientations when only one orientation is paired with
#     each image during training (as is the scenario for most pose estimation
#     datasets). In contrast to most pose estimation datasets, the full set of
#     equivalent orientations is available for evaluation.
    
#     There are eight shapes total, each rendered from 50,000 viewpoints distributed
#     uniformly at random over the full space of 3D rotations. Five of the shapes are
#     featureless -- tetrahedron, cube, icosahedron, cone, and cylinder. Of those, the
#     three Platonic solids (tetrahedron, cube, icosahedron) are annotated with their
#     12-, 24-, and 60-fold discrete symmetries, respectively. The cone and cylinder
#     are annotated with their continuous symmetries discretized at 1 degree
#     intervals. These symmetries are provided for evaluation; the intended
#     supervision is only a single rotation with each image.
    
#     The remaining three shapes are marked with a distinguishing feature. There is a
#     tetrahedron with one red-colored face, a cylinder with an off-center dot, and a
#     sphere with an X capped by a dot. Whether or not the distinguishing feature is
#     visible, the space of possible orientations is reduced. We do not provide the
#     set of equivalent rotations for these shapes.
    
#     Each example contains:
#     -   The 224x224 RGB image
#     -   A shape index for filtering by shape:
#         -   0 = tetrahedron
#         -   1 = cube
#         -   2 = icosahedron
#         -   3 = cone
#         -   4 = cylinder
#         -   5 = marked tetrahedron
#         -   6 = marked cylinder
#         -   7 = marked sphere
#     -   The rotation used in rendering (3x3 rotation matrix)
#     -   The set of known equivalent rotations under symmetry, for evaluation (if available).
# )

import tensorflow as tf
import tensorflow_datasets as tfds

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

import json
from PIL import Image
from tqdm import tqdm
import os
import cv2 as cv

class SymmetricSolidsDataset(Dataset):
    def __init__(self, split='train', transform=None, save=False):
        """
        Args:
            split (str): 'train', 'test'
            transform: 
        """
        self.transform = transform
        self.split = split

        self.data_dir = 'dataset'
        self.save_dir = os.path.join(self.data_dir, "symmetric_solids_dataset_customed", split)
        self.metadata = []

        if save:
            self.saveData(split=split)

        # load dataset
        if len(self.metadata) == 0:
            if split == "train":
                self.tf_dataset, self.info = tfds.load("symmetric_solids", split="train", with_info=True, data_dir=self.data_dir, as_supervised=True)
                # Iterate over dataset and save images/gt_label
                print("[INFO] Loading Training Data ...")
                for idx, (image, label) in tqdm(enumerate(self.tf_dataset.as_numpy_iterator())):              
                    label_matrix = label.tolist()  # Poise matrix as a nested list
                    image_path = os.path.join(self.save_dir, "images", f"{idx}.png")

                    # Append metadata
                    self.metadata.append({"image_path": image_path, "label": label_matrix})

                    # if idx == 1000:
                    #     break

            elif split == 'test':
                self.tf_dataset, self.info = tfds.load("symmetric_solids", split="test", with_info=True, data_dir=self.data_dir, as_supervised=False)
                # Iterate over dataset and save images/gt_label
                print("[INFO] Loading Testing Data ...")
                for idx, data in tqdm(enumerate(self.tf_dataset.as_numpy_iterator())):
                    # image = np.array(data["image"])
                    label_shape = np.array(data["label_shape"], dtype=np.int32)
                    rotation = np.array(data["rotation"])
                    rotations_equivalent = np.array(data["rotations_equivalent"])
                    
                    # Save the image as a PNG file
                    image_path = os.path.join(self.save_dir, "images", f"{idx}.png")

                    # Append metadata
                    self.metadata.append({"image_path": image_path, 
                                            "label_shape": label_shape.tolist(), 
                                            "rotation": rotation.tolist(), 
                                            "rotations_equivalent": rotations_equivalent.tolist()
                                            })
                    if idx == 2000:
                        break

    def saveData(self, split="train"):
        # Directory to save the dataset
        save_dir = os.path.join("dataset/symmetric_solids_dataset_customed", split)
        metadata_file = os.path.join(save_dir, "metadata.json")

        if True:
        # if not (os.path.exists(save_dir) and os.path.exists(metadata_file)):
            print(f"Create customed symsol dataset...")

            # Load the dataset from tensorflow_datasets (tfds)
            shapes = ["tet", "cube", "icosa", "cone", "cyl"]
            # Directory to save the dataset
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)

            if split == "train":
                tf_dataset, info = tfds.load("symmetric_solids", split="train", with_info=True, data_dir=self.data_dir, as_supervised=True)
                # Iterate over dataset and save images/gt_label
                for idx, (image, label) in tqdm(enumerate(tf_dataset.as_numpy_iterator())):
                # for idx, data in tqdm(enumerate(tf_dataset.as_numpy_iterator())):                
                    # Convert to numpy arrays
                    image = np.array(image)        # Image as Numpy array
                    label_matrix = label.tolist()  # Poise matrix as a nested list

                    # Save the image as a PNG file
                    image_path = os.path.join(save_dir, "images", f"{idx}.png")
                    Image.fromarray(np.array(image)).save(image_path)
                    
                    # Append metadata
                    self.metadata.append({"image_path": image_path, "label": label_matrix})

                # Save metadata to a JSON file
                with open(metadata_file, "w") as f:
                    json.dump(self.metadata, f, indent=4)

                print("Training dataset saved successfully!")
            elif split == "test":
                # as_supervised=False mean the load command would return all the features
                tf_dataset, info = tfds.load("symmetric_solids", split="test", with_info=True, data_dir=self.data_dir, as_supervised=False)

                for idx, data in tqdm(enumerate(tf_dataset.as_numpy_iterator())):
                    image = np.array(data["image"])
                    label_shape = np.array(data["label_shape"], dtype=np.int32)
                    rotation = np.array(data["rotation"])
                    rotations_equivalent = np.array(data["rotations_equivalent"])
                    
                    # Save the image as a PNG file
                    image_path = os.path.join(save_dir, "images", f"{idx}.png")
                    Image.fromarray(np.array(image)).save(image_path)

                    # Append metadata
                    self.metadata.append({"image_path": image_path, 
                                          "label_shape": label_shape.tolist(), 
                                          "rotation": rotation.tolist(), 
                                          "rotations_equivalent": rotations_equivalent.tolist()
                                          })
                
                # Save metadata to a JSON file
                with open(metadata_file, "w") as f:
                    json.dump(self.metadata, f, indent=4)

                print("Testing dataset saved successfully!")

    def __len__(self):
        return len(self.metadata)
        # return 2000
    
    def __getitem__(self, idx):
        """
        Returns:
        images (torch.Tensor): A tensor of stacked images with shape (batch_size, H, W, C).
                                Each image is a tensor of type float32.
        rotations (torch.Tensor): A tensor of stacked rotations with shape (batch_size, 3, 3).
                                  Each rotation is a tensor of type float32.
        rotations_equivalent (list): A list of tensors, each with shape (N, 3, 3), where N is the variable number of 
                                     rotations for each sample. Each tensor is of type float32.
        """
        # get the sample from the dataset
        entry = self.metadata[idx]

        # Retrive image_path from entry
        image_path = entry["image_path"]

        # Retrive rotation and rotations_equivalent from entry
        rotation, rotations_equivalent = None, None
        rotation = (
            torch.tensor(entry["label"], dtype=torch.float32) 
            if self.split == 'train' 
            else torch.tensor(entry["rotation"], dtype=torch.float32)
        )
        
        if self.split == 'test':
            rotations_equivalent = torch.tensor(entry['rotations_equivalent'])

        # Load the image with PIL-library and convert it to a numpy array
        image = Image.open(image_path).convert("RGB")  # H, W, C
        image = np.array(image)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, rotation, rotations_equivalent

    def get_equivalent(self, idx):
        return self.metadata[idx]["rotations_equivalent"]
    def get_label(self, idx):
        return self.metadata[idx]["label_shape"]

def load_symmetric_solids_dataset(split='train', transform=None, save=False):
    return SymmetricSolidsDataset(split=split, transform=transform, save=save)

def getDataLoader(dataset, batch_size, shuffle=True, num_workers=0):
    def custom_collate_fn(batch):
        """
        Custom collate function to handle varying sizes of rotations_equivalent.
        """

        images = [item[0] for item in batch]  # Stack images
        rotations = [item[1] for item in batch]  # Stack rotations
        rotations_equivalent = None

        # Ensure the first elements are tensors
        assert isinstance(images[0], torch.Tensor), "The image must be a PyTorch tensor."
        assert isinstance(rotations[0], torch.Tensor), "The rotation must be a PyTorch tensor."

        # Let rotations_equivalent as a list for non-uniform sizes
        if batch[0][2] is not None:
            rotations_equivalent = [item[2] for item in batch]
        
        # Stack images and rotations into batched tensors
        images = torch.stack(images)
        rotations = torch.stack(rotations)

        return images, rotations, rotations_equivalent
    
    return torch.utils.data.DataLoader(dataset=dataset, 
                                       batch_size=batch_size, 
                                       shuffle=shuffle, 
                                       num_workers=num_workers, 
                                       collate_fn=custom_collate_fn
                                       )

# Main function to test the dataset loader
def main():
    print("Loading dataset...")
    mode = 'train'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
    ])
    dataset = load_symmetric_solids_dataset(split='test', transform=transform, save=False)
    # dataset = load_symmetric_solids_dataset(split=mode, transform=transform, save=False)
    print(f"Dataset has length {len(dataset)}")

    batch_size = 2
    train_loader = getDataLoader(dataset, batch_size, shuffle=True)

    # print(f"Dataset Info: {dataset.info}")
    print(f"Loading batches with batch size {batch_size}...")
    for i, (images, rotation, rotations_equivalent) in enumerate(train_loader):
        print(f"Batch {i}:")

        # Display the first image in the batch
        img, rotation = images[0].numpy(), rotation[0]
        if mode == 'train':
            print(rotation)
        elif mode == 'test':
            rotations_equivalent = rotations_equivalent[0]
            print(rotations_equivalent.shape)
            print(rotations_equivalent[:3])
        
        img = np.transpose(img, (1, 2, 0))
        img = img + 0.5

        # Convert image color
        img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imshow(f"Batch {i}: Image", img_bgr)

        # Wait for user input
        key = cv.waitKey(0)
        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord(' '):
            print("Next image...")
            cv.destroyAllWindows()

if __name__ == "__main__":
    main()