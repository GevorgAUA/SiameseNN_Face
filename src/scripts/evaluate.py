import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data_processing.load_data import load_dataset
from src.training.data_loader import SiameseDataset
from src.evaluation.metrics import compute_accuracy, find_best_threshold
from src.models.siamese_network import SiameseNetwork
from src.models.base_network import BaseNetwork

# Configuration
dataset_dir = "dataset/archive/train"  # Path to dataset
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "siamese_model.pth"  # Path to saved model checkpoint

# Step 1: Load the validation/test dataset
_, (val_paths, val_labels) = load_dataset(dataset_dir)  # Load validation set only

# Step 2: Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Step 3: Create the validation DataLoader
val_dataset = SiameseDataset(val_paths, val_labels, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Step 4: Load the trained model
base_network = BaseNetwork()
siamese_model = SiameseNetwork(base_network).to(device)
siamese_model.load_state_dict(torch.load(checkpoint_path))  # Load weights
siamese_model.eval()

# Step 5: Compute validation accuracy
distances = []
labels = []

with torch.no_grad():
    for img1, img2, label in val_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        embedding1, embedding2 = siamese_model(img1, img2)
        distance = siamese_model.compute_distance(embedding1, embedding2)
        distances.append(distance)
        labels.append(label)

# Convert to tensors for metric computation
distances = torch.cat(distances)
labels = torch.cat(labels)

# Calculate metrics
threshold, best_accuracy = find_best_threshold(distances, labels)
print(f"Best Threshold: {threshold:.2f}, Validation Accuracy: {best_accuracy:.4f}")
