import os
import shutil
import random


# The first step of cleaning the data is removing any irrelevant classes.
# For this project, this means the "Bangla", "Chinese" and "Symbols" classes.

def remove_unwanted_images(gt_file_path, image_folders_paths):
    """
    Remove images labeled as 'Bangla', 'Chinese', or 'Symbols' from all folders
    and update the gt.txt file in words_part_1.

    Parameters:
    gt_file_path (str): Path to the gt.txt file in words_part_1.
    image_folders_paths (list): List of paths to the image folders.
    """
    # Read gt.txt content
    with open(gt_file_path, 'r') as file:
        lines = file.readlines()

    images_to_remove = set()
    updated_lines = []
    for line in lines:
        image_name, language, _ = line.split(',', 2)
        if language in ["Bangla", "Chinese", "Symbols"]:
            images_to_remove.add(image_name)
            print(f"Scheduled removal of {image_name} labeled as {language}")
        else:
            updated_lines.append(line)

    # Remove images from all folders
    for folder_path in image_folders_paths:
        for image_name in images_to_remove:
            image_path = os.path.join(folder_path, image_name)
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Removed {image_path}")
            else:
                print(f"File {image_path} not found in {folder_path}")

    # Write the updated content back to gt.txt
    with open(gt_file_path, 'w') as file:
        file.writelines(updated_lines)


# Paths to gt.txt and the image folders, relative to the src folder
gt_file_path_local = '../dataset/words_part_1/gt.txt'
image_folders_paths_local = ['../dataset/words_part_1', '../dataset/words_part_2', '../dataset/words_part_3']

remove_unwanted_images(gt_file_path_local, image_folders_paths_local)


# The second step should focus on creating the dataset directory in a way that is easy for PyTorch to handle later.
# 1. Read the gt.txt file to map each image to its language.
# 2. Stratify the data into training (70%) and testing (30%) sets, ensuring each language category is
#    proportionally represented.
# 3. Create the necessary training and testing directories with language subdirectories.
# 4. Move the files into these directories accordingly.

def create_directory(path):
    """
    Create a directory at the specified path if it does not exist.

    This function checks if a directory exists at the given path, and if not,
    it creates the directory including all intermediate directories necessary.

    Parameters:
    path (str): A string specifying the path where the directory should be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def read_language_mapping(gt_file_path):
    """
    Read the language mapping from the gt.txt file.

    Parameters:
    gt_file_path (str): Path to the gt.txt file.

    Returns:
    dict: A dictionary mapping image filenames to their respective languages.
    """
    mapping = {}
    with open(gt_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',', 2)
            if len(parts) == 3:
                image_name, language, _ = parts
                mapping[image_name] = language.lower()
    return mapping


def split_dataset(base_dir, gt_file_path, train_ratio=0.7):
    """
    Split the dataset into training and testing sets based on the specified ratio.

    This function reads the language information from the gt.txt file and performs
    a stratified split of the images into training and testing sets. It creates 'training'
    and 'testing' subdirectories within the base directory and moves the files into
    appropriate language-specific subdirectories within these.

    Parameters:
    base_dir (str): The base directory where the image files are stored.
    gt_file_path (str): Path to the gt.txt file containing language labels.
    train_ratio (float, optional): The ratio of data to be used for training. Default is 0.7.
    """
    # Read language mapping from gt.txt
    language_mapping = read_language_mapping(gt_file_path)

    # Initialize data structure for stratified split
    languages = set(language_mapping.values())
    data = {lang: [] for lang in languages}

    # Loop through each part directory and process image files
    for part_dir in ['words_part_1', 'words_part_2', 'words_part_3']:
        part_path = os.path.join(base_dir, part_dir)
        for image_name in os.listdir(part_path):
            if image_name in language_mapping:
                language = language_mapping[image_name]
                data[language].append(os.path.join(part_path, image_name))

    # Create training and testing directories
    train_dir = os.path.join(base_dir, 'training')
    test_dir = os.path.join(base_dir, 'testing')
    create_directory(train_dir)
    create_directory(test_dir)

    # Perform stratified split and move files
    for lang, files in data.items():
        print(f"Processing language: {lang}, Number of files: {len(files)}")  # Debugging
        random.shuffle(files)
        split_point = int(len(files) * train_ratio)
        train_files = files[:split_point]
        test_files = files[split_point:]

        # Create language specific directories
        train_lang_dir = os.path.join(train_dir, lang)
        test_lang_dir = os.path.join(test_dir, lang)
        create_directory(train_lang_dir)
        create_directory(test_lang_dir)

        # Move files
        for file in train_files:
            shutil.move(os.path.join(base_dir, file), train_lang_dir)
        for file in test_files:
            shutil.move(os.path.join(base_dir, file), test_lang_dir)


base_dataset_dir = '../dataset'
split_dataset(base_dataset_dir, gt_file_path_local)
