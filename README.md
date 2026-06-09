# **PokéNet: Building a Pokémon Image Classifier with ResNet50**

## **:round_pushpin:  Purpose**
For my DSCI 410L final project, I will be developing an image classifier that seeks to accurately classify a Pokémon based on a given image. Some images of the same Pokémon may have different qualities or art styles, so I am curious whether a classifier built using a convolutional neural network (CNN) can bypass these differences and accurately label the Pokémon.

Growing up, Pokémon has been one of my favorite hobbies. I chose it as the center of my term-long project so I could actively apply the skills acquired from a deep learning course to one of my interests. I hope that this project can mark the beginning of my interest in working on machine learning and deep learning projects outside of the classroom, where I can simultaneously explore my interests and further develop my coding and analytical skills.

## **:floppy_disk:  Data Overview**
The dataset I will be using for this project is titled [**"1000 Pokemon Dataset"**](https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000/data) from Kaggle. The dataset contains ~40 images per 1,000 Pokémon species, totaling over 26,000 images and 1,000 classes. The dataset has already been split into an 80% training set, 10% evaluation set, and 10% test set, so no additional splitting is required. An example notebook with data loaders is located under ```notebooks/data_demo.ipynb```.

## **:electric_plug:  Installation & Setup**
(OPTIONAL) Set up a virtual environment.
```
bash
python3 -m venv venv
source venv/bin/activate
```

Install the package.
```
git clone https://github.com/mandyzhan05/pokenet-image-classifier.git
cd pokenet-image-classifier
pip install -e .
```

### Data Preparation ###
1. Install Kagglehub.
``` pip install kagglehub ```

2. Authenticate with your Kaggle API token.
```
mkdir -p ~/.kaggle
echo [YOUR_TOKEN] > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

4. Set the dataset path environment variable.
``` export POKEMON_DATASET_PATH = $HOME/.cache/kagglehub/datasets/noodulz/pokemon-dataset-1000/versions/2/pokemon-dataset-1000 ```

## **:thought_balloon: Training **
1. Submit a batch job on Talapas. Otherwise, run and train the model directly.
``` sbatch scripts/train.sbatch ```
``` python train_model.py --config config.yaml ```

2. To modify parameters, edit ```config.yaml```:
```
epochs_phase1: 5
epochs_phase2: 10
learning_rate_phase1: 0.001
learning_rate_phase2: 0.00001
batch_size: 64
```

## **:package: Package Usage**
``` import os
from pokenet.dataloader import PokemonDataset
from torch.utils.data import DataLoader

train_dataset = PokemonDataset(os.environ["POKEMON_DATASET_PATH"])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

## **:clipboard: Results**

## **:thought_balloon: Limitations**