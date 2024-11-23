class EarlyStopping:
    def __init__(self, patience=5, delta=0.01):
        """
        Early stopping to terminate training when validation loss stops improving.
        :param patience: Number of epochs to wait before stopping.
        :param delta: Minimum change in the monitored metric to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, current_loss):
        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
