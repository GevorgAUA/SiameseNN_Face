import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        Contrastive loss for Siamese network training.
        :param margin: Margin for the dissimilar pairs.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        """
        Compute the contrastive loss.
        :param distance: Euclidean distance between embeddings.
        :param label: 1 for similar pairs, 0 for dissimilar pairs.
        :return: Contrastive loss value.
        """
        loss_similar = label * torch.pow(distance, 2)
        loss_dissimilar = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return torch.mean(loss_similar + loss_dissimilar)
