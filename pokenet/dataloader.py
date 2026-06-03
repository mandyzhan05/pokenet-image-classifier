# Imports
import torch
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

image_extensions = {".jpeg", ".jpg", ".png"}

class PokemonDataset(Dataset):
    '''
    Create a custom dataloader from the following Kaggle dataset: https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000/data.
    The dataset structure is based on images organized in folders by label
    '''

    def __init__(self, root_directory, transform = None):
        '''
        Args:
        root_directory: path to train/, val/, or test/ folders
        '''
        
        self.root_directory = Path(root_directory)
        self.transform = transform
        self.samples = []  # Initialize an empty list to hold the labels and corresponding images
        self.species_to_index = {}

        # Iterate through each folder and find the corresponding labels
        for label_index, species_directory in enumerate(sorted(self.root_directory.iterdir())):
            if species_directory.is_dir():
                species_name = species_directory.name
                self.species_to_index[species_name] = label_index
                for image_file in species_directory.iterdir():
                    if image_file.suffix.lower() in image_extensions:
                        self.samples.append((image_file, label_index))

    def __len__(self):
        '''
        Returns the number of samples
        '''
        
        return len(self.samples)

    def __getitem__(self, index):
        '''
        index = corresponding label
        '''
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        # Run any transformations on the image
        if self.transform:
            image = self.transform(image)

        # Return the transformed image and label as tensors
        label = torch.tensor(label, dtype = torch.long)

        return image, label

def data_transform(image_size = 256):
    '''
    Perform transformations to reduce dimensionality and create a single preprocessing pipeline
    '''

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
    ])  # Normalize

def get_data_loaders(base_path, batch_size = 64, image_size = 256):
    '''
    Creates data loaders for train, validation, and test sets

    Args:
    base_path (str): Base path to the dataset
    batch_size (int): Batch size for the dataloaders
    '''
    transform = data_transform(image_size)

    # Apply transformations to the dataset from each directory
    train_dataset = PokemonDataset(os.path.join(path, "train"), transform = transform)
    validation_dataset = PokemonDataset(os.path.join(path, "val"), transform = transform)
    test_dataset = PokemonDataset(os.path.join(path, "test"), transform = transform)

    # Get a batch of data of training data
    train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)
    validation_loader = DataLoader(dataset = validation_dataset, batch_size = 64)
    test_loader = DataLoader(dataset = test_dataset, batch_size = 64)