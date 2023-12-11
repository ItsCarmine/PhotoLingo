# PhotoLingo
 ECE 539 Fall 2023 Project at UW-Madison: CNN-based Language Identification from Images

# Project Overview
This project, developed for ECE 539 at the University of Wisconsin-Madison during the Fall 2023 semester, employs Convolutional Neural Networks (CNNs) to identify languages from images containing text of a single-script. It focuses on five languages: Arabic, Hindi, English (Latin script), Japanese, and Korean.

Team Members:
- Anais Corona Perez (CoronaPerez@wisc.edu)
- Carmine Shorette (CShorette@wisc.edu)
- Kentaro Takahashi (Takahashi5@wisc.edu)
- Amelia Zanin (AZanin@wisc.edu)

# Dataset
Our project utilizes the dataset from the ICDAR 2019 Robust Reading Challenge on Multilingual Scene Text Detection and Recognition, comprising 80,000 high resolution images of cropped text. The dataset is diverse, covering languages such as Latin, Japanese, Arabic, Korean and Hindi, with accompanying ground truth labels for each image. The dataset can be found [here](https://rrc.cvc.uab.es/?ch=15&com=downloads), but an account is required to view & download the data. You do not need all the data found on the website! This project only utilized the Task 2 dataset.

# Directory Organization
**If you wish to run the model on a local machine, your data must be structured like the `dataset` folder seen below.** Please see [Using Utilities](#utilities) instructions to setup the dataset directory. We also provide info on what you can expect to find in other folders.
```
PhotoLingo/
│
├── dataset/
│   ├── training/
│   │   ├── arabic/
│   │   ├── hindi/
│   │   ├── latin/
│   │   ├── japanese/
│   │   └── korean/
│   │
│   └── testing/
│       ├── arabic/
│       ├── hindi/
│       ├── latin/
│       ├── japanese/
│       └── korean/
│
├── documents/           # Project proposals, presentations, reports, etc.
│
├── src/                 # Source code for the CNN models and utilities
│
├── models/              # Trained model files and model architecture
│
├── notebooks/           # Jupyter notebooks for demos and experiments
│
├── requirements.txt     # List of dependencies for the project
│
├── LICENSE.txt          # Standard MIT license information
│
└── README.md            # Project documentation
```

# <a id="utilities"></a> Using Utilities

### Setting up the `Dataset` folder.
If you downloaded the data from the ICDAR website (specifically for task 2!), please unzip all 3 files in the `dataset` folder (not included upon cloning project, please create). Then run `data-setup.py` located in the `src` folder. This should setup the class labels and folder structure correctly.

# Reproducibility
After setting up the dataset in the correct order, one can easily load any of the .pth files included in the `models` folder and test the data. For example, to load the model and run it on the testing set:
```python
# Necessary imports
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os
from PIL import Image
import torch.nn.functional as F

# Define the transform (Changes based on what model you are running)
transform = transforms.Compose([
    transforms.Resize(256),              # Resize to 256x256
    transforms.CenterCrop(224),          # Crop to 224x224
    transforms.ToTensor(),               # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet standards
                         std=[0.229, 0.224, 0.225])
])

# Load the testing datasets
test_dataset = datasets.ImageFolder(root='../dataset/testing', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Assuming we are trying to test ResNet18 then the class must be defined
model = models.resnet18(weights=True)

# Load the saved model state
model.load_state_dict(torch.load('../models/PhotoLingo_ResNet18_v1.pth'))

# Put the model in evaluation mode if you are doing inference
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total}%')

```

# Methods & Results
Detailed explanation of processes and results conducted in this machine learning experiment can be found in `documents/PhotoLingo_Final_Report.pdf`. 
