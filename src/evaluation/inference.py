import torch
from PIL import Image
from src.data_processing.preprocess import Preprocessor
from src.data_processing.augment import Augmentor

class InferencePipeline:
    def __init__(self, model, preprocessor, augmentor=None, device="cpu"):
        """
        Initialize inference pipeline.
        :param model: Trained Siamese network model.
        :param preprocessor: Preprocessor instance for face detection and alignment.
        :param augmentor: Optional augmentor instance for applying augmentations.
        :param device: Device to run inference on (CPU/GPU).
        """
        self.model = model
        self.preprocessor = preprocessor
        self.augmentor = augmentor
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image_path):
        """
        Preprocess an image for inference.
        :param image_path: Path to the image file.
        :return: Preprocessed image tensor or None if preprocessing fails.
        """
        preprocessed = self.preprocessor.preprocess(image_path)
        if preprocessed is None:
            return None
        if self.augmentor:
            preprocessed = self.augmentor.augment(preprocessed)
        return torch.tensor(preprocessed).permute(2, 0, 1).unsqueeze(0).float()

    def infer(self, image_path1, image_path2):
        """
        Run inference on a pair of images to compute their similarity.
        :param image_path1: Path to the first image.
        :param image_path2: Path to the second image.
        :return: Distance between embeddings or None if preprocessing fails.
        """
        img1 = self.preprocess_image(image_path1)
        img2 = self.preprocess_image(image_path2)

        if img1 is None or img2 is None:
            print(f"Failed to preprocess one or both images: {image_path1}, {image_path2}")
            return None

        img1, img2 = img1.to(self.device), img2.to(self.device)

        with torch.no_grad():
            embedding1, embedding2 = self.model(img1, img2)
            distance = self.model.compute_distance(embedding1, embedding2)
        return distance.item()
