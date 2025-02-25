"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return nn.CrossEntropyLoss()(logits, target)


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()

        # create a linear layer
        self.linear = nn.Linear(h * w * 3, num_classes)
        self.loss_function = ClassificationLoss()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Get the batch size
        batch_size = x.size(0)

        x_flattened = x.reshape(batch_size, -1)

        logits = self.linear(x_flattened)

        return logits


class MLPClassifier(nn.Module):
    def __init__(
            self,
            h: int = 64,
            w: int = 64,
            num_classes: int = 6,
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()

        # Calculate input dimensions (flattened image)
        input_dim = 3 * h * w  # 3 channels * height * width
        hidden_dim = 64  # Size of the hidden layer

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x) # activation function

        return self.fc2(x)

class MLPClassifierDeep(nn.Module):
    def __init__(
            self,
            h: int = 64,
            w: int = 64,
            num_classes: int = 6,
            num_layers: int = 4,
            hidden_dim: int = 64
    ):
        """
        An MLP with multiple hidden layers using ModuleList for better layer management.

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int, number of output classes
            num_layers: int, number of hidden layers
            hidden_dim: int, size of hidden layers
        """
        super().__init__()

        input_dim = 3 * h * w

        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, num_classes)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        x = self.flatten(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)

        x = self.output_layer(x)

        return x

class MLPClassifierDeepResidual(nn.Module):
    def __init__(
            self,
            h: int = 64,
            w: int = 64,
            num_classes: int = 6,
            hidden_dim: int = 64,
            num_blocks: int = 5 # note that these are 2x a single hidden layer essentially
    ):
        """
        An MLP with residual connections between blocks using ModuleList

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int, number of output classes
            hidden_dim: int, size of hidden layers
            num_blocks: int, number of residual blocks
        """
        super().__init__()

        input_dim = 3 * h * w

        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        self.input_relu = nn.ReLU()

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        x = self.flatten(x)
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = self.input_relu(x)

        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        """
        A residual block for MLPs

        Args:
            hidden_dim: int, dimension of hidden layers
            dropout_rate: float, dropout probability
        """
        super().__init__()
        dropout_rate = 0.1

        # Main branch modules
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # Batch normalization normalizes the output of the previous layer to reduce noise
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Final activation after addition
        self.relu_out = nn.ReLU()
        self.dropout_out = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection

        Args:
            x: tensor (b, hidden_dim)

        Returns:
            tensor (b, hidden_dim)
        """
        identity = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out += identity

        out = self.relu_out(out)
        return self.dropout_out(out)


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
