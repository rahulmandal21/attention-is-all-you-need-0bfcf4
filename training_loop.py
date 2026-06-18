import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from typing import Tuple

class TransformerModel(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize the transformer model.

        Args:
        d_model (int): The dimension of the model.
        num_heads (int): The number of attention heads.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer model.

        Args:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor.
        """
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initialize the label smoothing loss function.

        Args:
        num_classes (int): The number of classes.
        smoothing (float, optional): The smoothing factor. Defaults to 0.1.
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.num_classes = num_classes

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the label smoothing loss function.

        Args:
        predictions (torch.Tensor): The predicted tensor.
        targets (torch.Tensor): The target tensor.

        Returns:
        torch.Tensor: The loss tensor.
        """
        return self.criterion(predictions, targets)


class TrainingLoop:
    def __init__(self, model: nn.Module, dataloader: DataLoader, d_model: int, warmup_steps: int, max_grad_norm: float = 1.0):
        """
        Initialize the training loop.

        Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): The data loader.
        d_model (int): The dimension of the model.
        warmup_steps (int): The number of warmup steps.
        max_grad_norm (float, optional): The maximum gradient norm. Defaults to 1.0.
        """
        self.model = model
        self.dataloader = dataloader
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.optimizer = Adam(model.parameters(), lr=1e-4)
        self.scheduler = LambdaLR(self.optimizer, self.lr_schedule)

    def lr_schedule(self, step: int) -> float:
        """
        The learning rate schedule.

        Args:
        step (int): The current step.

        Returns:
        float: The learning rate.
        """
        return self.d_model ** -0.5 * min(step ** -0.5, step * self.warmup_steps ** -1.5)

    def train_one_epoch(self) -> float:
        """
        Train the model for one epoch.

        Returns:
        float: The average loss.
        """
        self.model.train()
        total_loss = 0.0
        for batch in self.dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), targets.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = LabelSmoothingLoss(num_classes=10)(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
        return total_loss / len(self.dataloader)


if __name__ == "__main__":
    model = TransformerModel(d_model=512, num_heads=8)
    dataloader = DataLoader(torch.randn(100, 10, 512), batch_size=10)
    training_loop = TrainingLoop(model, dataloader, d_model=512, warmup_steps=1000)
    for epoch in range(10):
        loss = training_loop.train_one_epoch()
        print(f"Epoch {epoch+1}, Loss: {loss}")