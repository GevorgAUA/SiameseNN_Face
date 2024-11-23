import os
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data_processing.load_data import load_dataset
from src.training.data_loader import SiameseDataset
from src.models.base_network import BaseNetwork
from src.models.siamese_network import SiameseNetwork
from src.models.loss_functions import ContrastiveLoss
from src.training.trainer import Trainer
import torch
from torch import optim

# Configuration
dataset_dir = "dataset/archive/train"  # Path to dataset
batch_size = 32
epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 1: Load and split the dataset
(train_paths, train_labels), (val_paths, val_labels) = load_dataset(dataset_dir)

# Step 2: Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to standard dimensions
    transforms.ToTensor()
])
# Step 3: Create datasets and DataLoaders
train_dataset = SiameseDataset(train_paths, train_labels, transform=transform)
val_dataset = SiameseDataset(val_paths, val_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Step 4: Initialize the model, loss function, and optimizer
base_network = BaseNetwork()
siamese_model = SiameseNetwork(base_network).to(device)
criterion = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(siamese_model.parameters(), lr=0.001)

# Step 5: Train the model
trainer = Trainer(
    model=siamese_model,
    criterion=criterion,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device
)
trainer.fit(epochs=epochs)
