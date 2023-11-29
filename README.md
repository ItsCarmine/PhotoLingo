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

# Methods
To be continued...

# Results
**Reproducibility Note: Your results may be different, since the partitioning of the training and testing sets are completely random.**
Hopefully good...
