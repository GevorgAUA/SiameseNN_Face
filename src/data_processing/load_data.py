import os
from glob import glob
from sklearn.model_selection import train_test_split


def load_dataset(dataset_dir, train_split=0.96):
    """
    Load the dataset and split it into training and validation sets.
    :param dataset_dir: Path to the dataset directory (e.g., 'dataset/archive/train').
    :param train_split: Fraction of the data to use for training (default: 96%).
    :return: (train_paths, train_labels), (val_paths, val_labels)
    """
    image_paths = []
    labels = []

    # Map folder names to unique numerical labels
    folder_names = sorted(os.listdir(dataset_dir))  # Ensure consistent ordering
    for label, folder in enumerate(folder_names):
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            images = glob(os.path.join(folder_path, "*.jpg"))
            image_paths.extend(images)
            labels.extend([label] * len(images))

    # Split folders into training and validation sets
    num_train_folders = int(train_split * len(folder_names))
    train_folders = folder_names[:num_train_folders]
    val_folders = folder_names[num_train_folders:]

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    # Assign images to train/val splits based on folder
    for path, label in zip(image_paths, labels):
        folder_name = os.path.basename(os.path.dirname(path))
        if folder_name in train_folders:
            train_paths.append(path)
            train_labels.append(label)
        elif folder_name in val_folders:
            val_paths.append(path)
            val_labels.append(label)

    return (train_paths, train_labels), (val_paths, val_labels)
