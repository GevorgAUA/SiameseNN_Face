import torch


def compute_accuracy(distances, labels, threshold=0.5):
    """
    Calculate accuracy for validation by comparing distances with a threshold.
    :param distances: Tensor of distances between embeddings.
    :param labels: Ground truth labels (1 for similar, 0 for dissimilar).
    :param threshold: Distance threshold for classification.
    :return: Accuracy as a float.
    """
    predictions = (distances < threshold).float()
    correct = (predictions == labels).sum().item()
    return correct / len(labels)


def find_best_threshold(distances, labels, thresholds=None):
    """
    Find the optimal threshold that maximizes accuracy.
    :param distances: Tensor of distances between embeddings.
    :param labels: Ground truth labels.
    :param thresholds: List of thresholds to test.
    :return: Optimal threshold and corresponding accuracy.
    """
    if thresholds is None:
        thresholds = torch.arange(0, 2, 0.01)  # Range [0, 2] with step 0.01
    best_threshold = 0.5
    best_accuracy = 0

    for threshold in thresholds:
        accuracy = compute_accuracy(distances, labels, threshold)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold.item(), best_accuracy
