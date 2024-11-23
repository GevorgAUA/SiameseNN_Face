import torch
from src.data_processing.preprocess import Preprocessor
from src.data_processing.augment import Augmentor
from src.evaluation.inference import InferencePipeline
from src.models.siamese_network import SiameseNetwork
from src.models.base_network import BaseNetwork

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "siamese_model.pth"  # Path to saved model checkpoint
image1_path = "path_to_image1.jpg"  # Replace with actual path
image2_path = "path_to_image2.jpg"  # Replace with actual path

# Step 1: Initialize Preprocessor and Augmentor
preprocessor = Preprocessor(image_size=(224, 224))
augmentor = Augmentor()

# Step 2: Load the trained model
base_network = BaseNetwork()
siamese_model = SiameseNetwork(base_network).to(device)
siamese_model.load_state_dict(torch.load(checkpoint_path))
siamese_model.eval()

# Step 3: Run inference
pipeline = InferencePipeline(model=siamese_model, preprocessor=preprocessor, augmentor=augmentor, device=device)
distance = pipeline.infer(image1_path, image2_path)

# Step 4: Output the result
if distance is not None:
    print(f"Similarity Distance: {distance:.4f}")
else:
    print("Failed to process one or both images.")
