import torch
import torch.nn as nn

class CrossEntropyLossFunction(nn.Module):
    """
    A PyTorch module that calculates the cross-entropy loss for sequence-to-sequence models.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1) -> None:
        """
        Initializes the CrossEntropyLossFunction module.

        Args:
        - num_classes (int): The number of classes in the classification problem.
        - smoothing (float, optional): The label smoothing factor. Defaults to 0.1.
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing, reduction='mean')
        self.num_classes = num_classes

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cross-entropy loss between the predictions and targets.

        Args:
        - predictions (torch.Tensor): The predicted logits.
        - targets (torch.Tensor): The true labels.

        Returns:
        - torch.Tensor: The cross-entropy loss.
        """
        return self.criterion(predictions, targets)


if __name__ == "__main__":
    # Create a dummy dataset
    predictions = torch.randn(10, 5, 10)  # batch_size, sequence_length, num_classes
    targets = torch.randint(0, 10, (10, 5))  # batch_size, sequence_length

    # Initialize the loss function
    loss_fn = CrossEntropyLossFunction(num_classes=10)

    # Calculate the loss
    loss = loss_fn(predictions, targets)

    print(f"Loss: {loss.item()}")