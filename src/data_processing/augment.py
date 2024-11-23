import random
from torchvision import transforms
from PIL import Image
import numpy as np

class Augmentor:
    def __init__(self):
        """
        Initialize augmentation pipelines with probabilistic application.
        """
        self.augmentations = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Brightness and contrast
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),  # Blur
        ])

    def augment(self, image):
        """
        Apply augmentations to an input image with set probabilities.
        :param image: Input image as a NumPy array or PIL image.
        :return: Augmented image.
        """
        image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        return self.augmentations(image)
