import torch
from tqdm import tqdm

from src.evaluation.metrics import compute_accuracy, find_best_threshold


class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, device):
        """
        Initialize the training pipeline.
        :param model: The Siamese network model.
        :param criterion: Loss function (e.g., ContrastiveLoss).
        :param optimizer: Optimizer (e.g., Adam).
        :param train_loader: DataLoader for training data.
        :param val_loader: DataLoader for validation data.
        :param device: Device to run the training on (CPU/GPU).
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train_epoch(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        for img1, img2, labels in tqdm(self.train_loader, desc="Training"):
            img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            embedding1, embedding2 = self.model(img1, img2)
            distances = self.model.compute_distance(embedding1, embedding2)
            loss = self.criterion(distances, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        """
        Validate the model and compute accuracy and optimal threshold.
        """
        self.model.eval()
        distances, labels = [], []
        total_loss = 0

        with torch.no_grad():
            for img1, img2, label in tqdm(self.val_loader, desc="Validation"):
                img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)

                embedding1, embedding2 = self.model(img1, img2)
                distance = self.model.compute_distance(embedding1, embedding2)
                loss = self.criterion(distance, label)
                total_loss += loss.item()

                distances.append(distance)
                labels.append(label)

        distances = torch.cat(distances)
        labels = torch.cat(labels)

        # Calculate accuracy and find best threshold
        accuracy = compute_accuracy(distances, labels)
        best_threshold, best_accuracy = find_best_threshold(distances, labels)

        print(
            f"Validation Accuracy: {accuracy:.4f}, Best Threshold: {best_threshold:.2f}, Best Accuracy: {best_accuracy:.4f}")
        return total_loss / len(self.val_loader), accuracy

    def fit(self, epochs):
        """
        Fit the model for the specified number of epochs.
        """
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.validate()

            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
