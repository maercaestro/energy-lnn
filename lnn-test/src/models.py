from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from ncps.torch import CfC
except ImportError:  # pragma: no cover
    CfC = None


class PhysicsHead(nn.Module):
    def __init__(self, hidden_size: int, output_size: int = 2) -> None:
        super().__init__()
        self.net = nn.Linear(hidden_size, output_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class LNN(nn.Module):
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        phys_output_size: int = 2,
        mixed_memory: bool = True,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        if CfC is None:
            raise ImportError("ncps is required to use the LNN model.")

        self.cfc_body = CfC(
            input_size,
            hidden_size,
            mixed_memory=mixed_memory,
            batch_first=batch_first,
        )
        self.phys_head = PhysicsHead(hidden_size, phys_output_size)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_seq, last_h = self.cfc_body(x, hx)
        return self.phys_head(hidden_seq), last_h


class LSTMBaseline(nn.Module):
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        phys_output_size: int = 2,
        dropout: float = 0.1,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=batch_first,
        )
        self.phys_head = PhysicsHead(hidden_size, phys_output_size)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hidden_seq, last_h = self.lstm(x, hx)
        return self.phys_head(hidden_seq), last_h


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_size: int = 5,
        hidden_dims: Optional[list[int]] = None,
        phys_output_size: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        layers: list[nn.Module] = []
        last_dim = input_size
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(last_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, phys_output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, hx: None = None) -> Tuple[torch.Tensor, None]:
        batch_size, seq_len, feature_dim = x.shape
        flat = x.reshape(batch_size * seq_len, feature_dim)
        pred = self.net(flat).reshape(batch_size, seq_len, -1)
        return pred, None


def create_model(
    model_name: str,
    input_size: int,
    target_size: int,
    model_config: dict,
    device: str,
) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "lnn":
        model = LNN(
            input_size=input_size,
            hidden_size=model_config.get("hidden_size", 128),
            phys_output_size=target_size,
            mixed_memory=model_config.get("mixed_memory", True),
        )
    elif model_name == "lstm":
        model = LSTMBaseline(
            input_size=input_size,
            hidden_size=model_config.get("hidden_size", 128),
            num_layers=model_config.get("num_layers", 2),
            phys_output_size=target_size,
            dropout=model_config.get("dropout", 0.1),
        )
    elif model_name == "mlp":
        model = SimpleMLP(
            input_size=input_size,
            hidden_dims=model_config.get("hidden_dims", [128, 64]),
            phys_output_size=target_size,
            dropout=model_config.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model.to(device)
