import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np


class SiameseDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Dataset for dynamically generating pairs of images.
        :param image_paths: List of paths to images.
        :param labels: Corresponding labels for the images.
        :param transform: Transformations to apply (including augmentation).
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_indices = self._group_by_class()

    def _group_by_class(self):
        """
        Group image indices by their labels for efficient pair generation.
        :return: Dictionary mapping labels to lists of indices.
        """
        class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def __len__(self):
        # Dataset size is the number of pairs we want to generate
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Generate a pair of images: either positive (same class) or negative (different class).
        """
        is_positive = random.choice([True, False])
        img1_path = self.image_paths[index]
        label1 = self.labels[index]

        if is_positive:
            # Positive pair: Same class
            idx2 = random.choice(self.class_indices[label1])
            while idx2 == index:
                idx2 = random.choice(self.class_indices[label1])
        else:
            # Negative pair: Different class
            other_labels = list(self.class_indices.keys())
            if len(other_labels) > 1:
                other_labels.remove(label1)
                label2 = random.choice(other_labels)
                idx2 = random.choice(self.class_indices[label2])
            else:
                # Fallback: Create a positive pair if no negative pair is possible
                idx2 = random.choice(self.class_indices[label1])
                while idx2 == index:
                    idx2 = random.choice(self.class_indices[label1])

        img2_path = self.image_paths[idx2]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor([1 if is_positive else 0], dtype=torch.float32)
        return img1, img2, label
