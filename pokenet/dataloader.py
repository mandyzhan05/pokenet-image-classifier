# Imports
import torch
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

image_extensions = {".jpeg", ".jpg", ".png"}

# train_directory = os.path.join(path, "pokemon-dataset-1000", "train")

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