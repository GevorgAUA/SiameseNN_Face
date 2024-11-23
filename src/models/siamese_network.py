import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self, base_network):
        """
        Siamese network composed of two identical base networks.
        :param base_network: An instance of the feature extractor.
        """
        super(SiameseNetwork, self).__init__()
        self.base_network = base_network

    def forward(self, input1, input2):
        """
        Forward pass through the Siamese network.
        :param input1: First input image.
        :param input2: Second input image.
        :return: Distance between embeddings of the two inputs.
        """
        embedding1 = self.base_network(input1)
        embedding2 = self.base_network(input2)
        return embedding1, embedding2

    def compute_distance(self, embedding1, embedding2):
        """
        Compute the Euclidean distance between two embeddings.
        :param embedding1: Embedding from the first branch.
        :param embedding2: Embedding from the second branch.
        :return: Distance between embeddings.
        """
        return F.pairwise_distance(embedding1, embedding2)
