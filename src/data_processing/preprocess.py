import cv2
import numpy as np
from mtcnn import MTCNN


class Preprocessor:
    def __init__(self, image_size=(224, 224)):
        """
        Initialize the preprocessor with a target image size and MTCNN detector.
        """
        self.image_size = image_size
        self.detector = MTCNN()

    def detect_and_align(self, image_path):
        """
        Detect and align the face in the image using MTCNN.
        :param image_path: Path to the input image.
        :return: Aligned and cropped face image or None if no face is detected.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        detections = self.detector.detect_faces(image)

        if not detections:
            print(f"No face detected in image: {image_path}")
            return None

        # Assume the largest detected face is the primary face
        detection = max(detections, key=lambda x: x['box'][2] * x['box'][3])
        x, y, w, h = detection['box']

        # Crop and align
        cropped_face = image[y:y+h, x:x+w]
        aligned_face = cv2.resize(cropped_face, self.image_size)
        return aligned_face

    def normalize(self, image):
        """
        Normalize the image to have pixel values in the range [0, 1].
        :param image: Input image as a NumPy array.
        :return: Normalized image.
        """
        image = np.array(image, dtype=np.float32) / 255.0
        return image

    def preprocess(self, image_path):
        """
        Full preprocessing pipeline: detect, align, and normalize.
        :param image_path: Path to the input image.
        :return: Preprocessed image or None if no face detected.
        """
        aligned_face = self.detect_and_align(image_path)
        if aligned_face is None:
            return None
        return self.normalize(aligned_face)
