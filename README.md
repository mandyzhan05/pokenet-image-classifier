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
```
export POKEMON_DATASET_PATH = $HOME/.cache/kagglehub/datasets/noodulz/pokemon-dataset-1000/versions/2/pokemon-dataset-1000
```

## **:thought_balloon: Training**
1. Submit a batch job on Talapas. Otherwise, run and train the model directly.
``` sbatch scripts/train.sbatch ```
``` python train_model.py --config config.yaml ```

2. To modify parameters, edit ```config.yaml```:
```
epochs_phase1: 10
epochs_phase2: 50
learning_rate_phase1: 0.001
learning_rate_phase2: 0.00001
batch_size: 64
```

Model weights are not included in this repository due to file size. To obtain the weights, run ``` python train_model.py --config configs/default.yaml ```.

## **:package: Package Usage**
``` import os
from pokenet.dataloader import PokemonDataset
from torch.utils.data import DataLoader

train_dataset = PokemonDataset(os.environ["POKEMON_DATASET_PATH"])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

## **:clipboard: Results**
During evaluation, I checked the performance on species classification and analyzed its ability to detect shiny versions. Training results after 10 and 50 epochs can be found under ``` results/ ```. All evaluation metrics were computed using the best model checkpoint (``` results/pokenet_best.pth ```), which saved at the epoch with the lowest validation loss during Phase 2 training rather than the final epoch weights (``` results/pokenet.pth ```). This ensured results reflected the model at its peak performance rather than risking overfitting at later epochs.

### Species Classification Metrics Comparisons ###
Macro F1 score was used to balance the normal-to-shiny ratio, and the top five accuracy calculates whether the model's prediction was in its top five guesses and the corresponding accuracy.

* Phase 1: 5 epochs | Phase 2: 10 epochs
    * Macro F1 score: 0.8181475690975692
    * Top 5 accuracy: 0.9194195739425749
* Phase 1: 10 epochs | Phase 2: 50 epochs
    * Macro F1 score: 0.9536894103118246
    * Top 5 accuracy: 0.8959444888444889
 
There was significant improvement after running additional epochs. Validation loss continued to improve even with 50 epochs during Phase 2 training. Early stopping was never triggered, suggesting that additional epochs could be beneficial in helping the model converge, potentially improving performance.

I also compared the performance of the final model checkpoint with the best model checkpoint, which revealed identical F1 and accuracy scores across the board. This indicates that the final model checkpoint was, in fact, the best model checkpoint. As mentioned earlier, additional epochs could be beneficial in helping the model converge, which could highlight more apparent differences between the model checkpoints.

### Confusion ###
<img width="1200" height="1000" alt="confusion_matrix 4 01 59 PM" src="https://github.com/user-attachments/assets/5827a5f1-462c-46ac-a784-893cbd03e31f" />

### Samples Prediction Visualization ###
<img width="1500" height="300" alt="sample_predictions" src="https://github.com/user-attachments/assets/c4bd8b22-11fe-42fd-845d-928562c7b93b" />

### Shiny Detection Analysis ###
To take my project a step further, I tested its ability to differentiate a Pokémon from its shiny version versus its original form. Since none of the shiny variations had ground truth labels indicating as such, I was unable to explicitly ask if a species was shiny or not. Instead, I manually identified the shiny versions first, then tested the model's confidence in predicting the species. My hypothesis was that the model was less confident when looking at the shiny version because the colors were different and would be higher if the versions were harder to distinguish.

<img width="990" height="495" alt="shiny_confidence 4 01 59 PM" src="https://github.com/user-attachments/assets/b7dd12f1-464b-4b57-8b9b-0326075c1c4d" />

The confidence gap in identifying Garchomp is slightly larger than in identifying Wooper. Across both Pokémon, the model appears to correctly identify the species the majority of the time, and confidence in identifying the shiny versions is slightly less than identifying normal versions.

## **:exclamation: Limitations & Future Changes**
Although there were 1000 classes, the dataset had around 21 training images per species and even fewer in the validation and test set. Furthermore, the dataset had more images and varying art styles for older species (Gen I-Gen VI) than newer ones with significantly different art styles (Gen VII-Gen X). There were also no ground truth labels separating normal and shiny versions, so I had to manually identify them, which was subject to higher human error for species with minimal differences between versions.

If I were to change my course of action, I would consider a different interpretation of shiny detection analysis by manually labeling a small subset of shiny images across species and train those labels, measuring how shiny detection improves with additional labels. It would also be interesting to observe regional variations and how the model adapts to those, since regional forms also have distinct features from their original form. Finally, using GradCAM for feature visualization could provide a better visualization of which parts of the image the model focuses on for normal vs. shiny versions.
