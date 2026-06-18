import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input tensor.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initializes the PositionalEncoding module.

        Args:
        d_model (int): The dimension of the input tensor.
        max_len (int): The maximum length of the input sequence.
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                self.pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                self.pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The input tensor with positional encoding.
        """
        return x + self.pe[:, :x.size(1), :]

class TokenizedTextDataset(Dataset):
    """
    A dataset class for tokenized text sequences.
    """
    def __init__(self, sequences: list):
        """
        Initializes the TokenizedTextDataset class.

        Args:
        sequences (list): A list of tokenized text sequences.
        """
        self.sequences = [torch.tensor(seq) for seq in sequences]

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        int: The length of the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns a tokenized text sequence at the given index.

        Args:
        idx (int): The index of the sequence.

        Returns:
        torch.Tensor: The tokenized text sequence at the given index.
        """
        return self.sequences[idx]

def collate_fn(batch: list) -> torch.Tensor:
    """
    A custom collate function for padding sequences.

    Args:
    batch (list): A list of sequences.

    Returns:
    torch.Tensor: The padded sequences.
    """
    return pad_sequence(batch, batch_first=True, padding_value=0)

class DataPreprocessing:
    """
    A class for data preprocessing.
    """
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000):
        """
        Initializes the DataPreprocessing class.

        Args:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimension of the input tensor.
        max_len (int): The maximum length of the input sequence.
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

    def preprocess(self, sequences: list) -> DataLoader:
        """
        Preprocesses the input sequences.

        Args:
        sequences (list): A list of tokenized text sequences.

        Returns:
        DataLoader: A DataLoader object for the preprocessed sequences.
        """
        dataset = TokenizedTextDataset(sequences)
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        return dataloader

    def embed(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Embeds the input sequence.

        Args:
        sequence (torch.Tensor): The input sequence.

        Returns:
        torch.Tensor: The embedded sequence.
        """
        embedded_sequence = self.embedding(sequence)
        return embedded_sequence

    def add_positional_encoding(self, embedded_sequence: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the embedded sequence.

        Args:
        embedded_sequence (torch.Tensor): The embedded sequence.

        Returns:
        torch.Tensor: The embedded sequence with positional encoding.
        """
        sequence_with_positional_encoding = self.positional_encoding(embedded_sequence)
        return sequence_with_positional_encoding

if __name__ == "__main__":
    vocab_size = 1000
    d_model = 512
    max_len = 5000
    data_preprocessing = DataPreprocessing(vocab_size, d_model, max_len)
    sequences = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    dataloader = data_preprocessing.preprocess(sequences)
    for batch in dataloader:
        embedded_batch = data_preprocessing.embed(batch)
        batch_with_positional_encoding = data_preprocessing.add_positional_encoding(embedded_batch)
        print(batch_with_positional_encoding.shape)
        break