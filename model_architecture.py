import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    """
    A PyTorch implementation of the Transformer model architecture.
    
    The Transformer model consists of an encoder and a decoder, each composed of a stack of identical layers.
    The encoder maps an input sequence to a sequence of continuous representations, and the decoder generates an output sequence one element at a time.
    The model uses multi-head self-attention and point-wise, fully connected layers for both the encoder and decoder.
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int, input_dim: int, output_dim: int):
        """
        Initializes the Transformer model.
        
        Args:
        d_model (int): The number of features in the input and output data.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of layers in the encoder and decoder.
        input_dim (int): The dimensionality of the input data.
        output_dim (int): The dimensionality of the output data.
        """
        super().__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, input_dim)
        self.decoder = TransformerDecoder(d_model, num_heads, num_layers, output_dim)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the Transformer model.
        
        Args:
        input_seq (torch.Tensor): The input sequence.
        
        Returns:
        torch.Tensor: The output sequence.
        """
        encoder_output = self.encoder(input_seq)
        decoder_output = self.decoder(encoder_output)
        return decoder_output


class TransformerEncoder(nn.Module):
    """
    A PyTorch implementation of the Transformer encoder.
    
    The Transformer encoder consists of a stack of identical layers, each composed of multi-head self-attention and point-wise, fully connected layers.
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int, input_dim: int):
        """
        Initializes the Transformer encoder.
        
        Args:
        d_model (int): The number of features in the input and output data.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of layers in the encoder.
        input_dim (int): The dimensionality of the input data.
        """
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads) for _ in range(num_layers)])

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the Transformer encoder.
        
        Args:
        input_seq (torch.Tensor): The input sequence.
        
        Returns:
        torch.Tensor: The output sequence.
        """
        input_seq = self.input_linear(input_seq)
        for layer in self.layers:
            input_seq = layer(input_seq)
        return input_seq


class TransformerEncoderLayer(nn.Module):
    """
    A PyTorch implementation of a single layer in the Transformer encoder.
    
    Each layer consists of multi-head self-attention and point-wise, fully connected layers.
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Initializes the Transformer encoder layer.
        
        Args:
        d_model (int): The number of features in the input and output data.
        num_heads (int): The number of attention heads.
        """
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, num_heads)
        self.linear = nn.Linear(d_model, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the Transformer encoder layer.
        
        Args:
        input_seq (torch.Tensor): The input sequence.
        
        Returns:
        torch.Tensor: The output sequence.
        """
        attn_output = self.self_attn(input_seq, input_seq)
        norm_output = self.layer_norm1(input_seq + attn_output[0])
        linear_output = self.linear(norm_output)
        output = self.layer_norm2(norm_output + linear_output)
        return output


class TransformerDecoder(nn.Module):
    """
    A PyTorch implementation of the Transformer decoder.
    
    The Transformer decoder consists of a stack of identical layers, each composed of multi-head self-attention and point-wise, fully connected layers.
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int, output_dim: int):
        """
        Initializes the Transformer decoder.
        
        Args:
        d_model (int): The number of features in the input and output data.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of layers in the decoder.
        output_dim (int): The dimensionality of the output data.
        """
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.output_linear = nn.Linear(d_model, output_dim)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the Transformer decoder.
        
        Args:
        input_seq (torch.Tensor): The input sequence.
        
        Returns:
        torch.Tensor: The output sequence.
        """
        for layer in self.layers:
            input_seq = layer(input_seq)
        output = self.output_linear(input_seq)
        return output


class TransformerDecoderLayer(nn.Module):
    """
    A PyTorch implementation of a single layer in the Transformer decoder.
    
    Each layer consists of multi-head self-attention and point-wise, fully connected layers.
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Initializes the Transformer decoder layer.
        
        Args:
        d_model (int): The number of features in the input and output data.
        num_heads (int): The number of attention heads.
        """
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, num_heads)
        self.linear = nn.Linear(d_model, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the Transformer decoder layer.
        
        Args:
        input_seq (torch.Tensor): The input sequence.
        
        Returns:
        torch.Tensor: The output sequence.
        """
        attn_output = self.self_attn(input_seq, input_seq)
        norm_output = self.layer_norm1(input_seq + attn_output[0])
        linear_output = self.linear(norm_output)
        output = self.layer_norm2(norm_output + linear_output)
        return output


if __name__ == "__main__":
    model = TransformerModel(d_model=512, num_heads=8, num_layers=6, input_dim=512, output_dim=512)
    input_seq = torch.randn(1, 10, 512)
    output = model(input_seq)
    print(output.shape)