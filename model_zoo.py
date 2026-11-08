"""Eleven network families for independent force and contact-location estimation.

Inputs have shape (batch, time, feature). Outputs are three force components or
contact-location logits for the final time step. MLP uses only the final row;
other models consume input histories. All temporal operations are causal.
Units, normalization and no-load labels are handled outside the models.

To modify a model, update DEFAULT_PARAMS, its encoder and _build_encoder.
Per-run sizes can be overridden with force_params/location_params in configs/train.yaml.
window_samples is a data setting; the default 30 rows is not an optimized length.
KalmanNet is a legacy name for a GRU+MLP model, not a physical Kalman filter.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# 1. Model families and default sizes.
MODEL_NAMES = (
    "mlp", "cnn", "convmixer", "resnet", "lstm", "gru", "tcn",
    "transformer", "kalmannet", "small_gru", "residual_tcn",
)
MODEL_DESCRIPTIONS = {
    "mlp": "MLP208/104/52 (final time step only)",
    "cnn": "Causal CNN64, kernel3 × 2",
    "convmixer": "Causal ConvMixer64, depthwise kernel5",
    "resnet": "Causal ResNet64, residual blocks × 2",
    "lstm": "LSTM60 → LSTM30 → Dense32",
    "gru": "GRU72 → GRU36",
    "tcn": "Causal TCN64, dilation1/2/4 (receptive field 15)",
    "transformer": "Transformer32, heads2, FF64, causal + sinusoidal position",
    "kalmannet": "GRU96 + MLP96 (legacy KalmanNetLite; not a Kalman filter)",
    "small_gru": "GRU64, 1 layer",
    "residual_tcn": "Residual causal TCN32, dilation1/2/4 (receptive field 29)",
}


DEFAULT_PARAMS = {
    "mlp": {"hidden_dims": [208, 104, 52]},
    "cnn": {"channels": 64, "kernel_size": 3, "num_layers": 2},
    "convmixer": {"channels": 64, "num_blocks": 1},
    "resnet": {"channels": 64, "num_blocks": 2},
    "lstm": {"hidden_sizes": [60, 30], "projection_dim": 32},
    "gru": {"hidden_sizes": [72, 36], "projection_dim": None},
    "tcn": {"channels": 64, "kernel_size": 3, "dilations": [1, 2, 4]},
    "transformer": {"d_model": 32, "nhead": 2, "dim_feedforward": 64,
                    "dropout": 0.1, "num_layers": 1},
    "kalmannet": {"hidden_sizes": [96], "projection_dim": 96},
    "small_gru": {"hidden_sizes": [64], "projection_dim": None},
    "residual_tcn": {"channels": 32, "dilations": [1, 2, 4]},
}


def _positive_int(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def resolve_params(name: str, params: Mapping | None = None) -> dict:
    """Validate per-family overrides and return a fresh, serializable mapping.

    Recurrent ``hidden_sizes`` specifies one width per recurrent layer; an
    optional ``projection_dim`` adds a final Linear/ReLU. Convolution blocks
    remain causal. Unknown keys raise instead of silently using defaults.
    """
    if not isinstance(name, str) or name.lower() not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    name = name.lower()
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise ValueError("params must be a mapping")
    unknown = set(params) - set(DEFAULT_PARAMS[name])
    if unknown:
        raise ValueError(f"Unknown {name} parameters: {sorted(unknown, key=str)}")
    resolved = copy.deepcopy(DEFAULT_PARAMS[name])
    resolved.update(copy.deepcopy(dict(params)))
    for key, value in resolved.items():
        if key in {"hidden_dims", "hidden_sizes", "dilations"}:
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f"{name}.{key} must be a nonempty list of positive integers")
            resolved[key] = [_positive_int(item, f"{name}.{key}") for item in value]
        elif key == "dropout":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError("transformer.dropout must satisfy 0 <= dropout < 1")
            resolved[key] = float(value)
        elif key == "projection_dim" and value is None:
            continue
        else:
            _positive_int(value, f"{name}.{key}")
    if name == "transformer" and resolved["d_model"] % resolved["nhead"]:
        raise ValueError("transformer.d_model must be divisible by nhead")
    return resolved


# 2. Shared convolution blocks.
# AUTHOR: Daeyun
# Project implementations and adaptations of the network families listed above.
class CausalConv1d(nn.Conv1d):
    """Use only current/past values in B,C,T inputs while preserving sequence length."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, groups: int = 1):
        super().__init__(in_channels, out_channels, kernel_size,
                         dilation=dilation, groups=groups, padding=0)
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each time step for B,C,T inputs."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.path = nn.Sequential(
            CausalConv1d(channels, channels, 3, dilation=dilation),
            nn.ReLU(),
            CausalConv1d(channels, channels, 3, dilation=dilation),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x + self.path(x))


class ConvMixerBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Sequential(
            CausalConv1d(channels, channels, 5, groups=channels),
            nn.GELU(), ChannelLayerNorm(channels),
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GELU(), ChannelLayerNorm(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(x + self.depthwise(x))


# 3. Encoders for each model family.
class ConvolutionEncoder(nn.Module):
    """Temporal CNN/ConvMixer/ResNet/TCN encoder with output shape (B,T,channels)."""

    def __init__(self, input_dim: int, name: str, params: dict):
        super().__init__()
        channels = params["channels"]
        self.output_dim = channels
        if name == "cnn":
            layers = []
            for index in range(params["num_layers"]):
                layers.extend([CausalConv1d(input_dim if index == 0 else channels,
                                           channels, params["kernel_size"]), nn.ReLU()])
        elif name == "convmixer":
            layers = [CausalConv1d(input_dim, channels, 3), nn.GELU(),
                      ChannelLayerNorm(channels)]
            layers.extend(ConvMixerBlock(channels) for _ in range(params["num_blocks"]))
        elif name == "resnet":
            layers = [CausalConv1d(input_dim, channels, 3), nn.ReLU()]
            layers.extend(ResidualBlock(channels) for _ in range(params["num_blocks"]))
        elif name == "tcn":
            layers = []
            for index, dilation in enumerate(params["dilations"]):
                layers.extend([CausalConv1d(input_dim if index == 0 else channels,
                                           channels, params["kernel_size"], dilation=dilation),
                               nn.ReLU()])
        else:  # residual_tcn; the existing residual block uses two kernel-3 convolutions.
            layers = [nn.Conv1d(input_dim, channels, 1), nn.ReLU()]
            layers.extend(ResidualBlock(channels, dilation=d) for d in params["dilations"])
        self.network = nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        return self.network(x.transpose(1, 2)).transpose(1, 2)


class RecurrentEncoder(nn.Module):
    """hidden_sizes sets LSTM/GRU widths and depth; projection_dim sets the final projection."""

    def __init__(self, input_dim: int, name: str, params: dict):
        super().__init__()
        recurrent = nn.LSTM if name == "lstm" else nn.GRU
        widths = params["hidden_sizes"]
        # Preserve first/second keys in existing checkpoints; custom sizes use layers.
        self.default_layout = params == DEFAULT_PARAMS[name]
        if self.default_layout:
            self.first = recurrent(input_dim, widths[0], batch_first=True)
            self.second = (recurrent(widths[0], widths[1], batch_first=True)
                           if len(widths) == 2 else None)
        else:
            self.layers = nn.ModuleList()
            width = input_dim
            for hidden_size in widths:
                self.layers.append(recurrent(width, hidden_size, batch_first=True))
                width = hidden_size
        projection_dim = params["projection_dim"]
        self.projection = (nn.Identity() if projection_dim is None else
                           nn.Sequential(nn.Linear(widths[-1], projection_dim), nn.ReLU()))
        self.output_dim = widths[-1] if projection_dim is None else projection_dim

    def forward(self, x: Tensor) -> Tensor:
        # Reset hidden state for each window; never carry state between windows.
        layers = ([self.first, self.second] if self.default_layout else self.layers)
        for layer in layers:
            if layer is not None:
                x, _ = layer(x)
        return self.projection(x)


class TransformerEncoder(nn.Module):
    """Transformer with positional encoding and a causal mask; future rows are hidden."""

    def __init__(self, input_dim: int, seq_len: int, params: dict):
        super().__init__()
        width = params["d_model"]
        self.output_dim = width
        self.projection = nn.Linear(input_dim, width)
        layers = [nn.TransformerEncoderLayer(
            d_model=width, nhead=params["nhead"],
            dim_feedforward=params["dim_feedforward"], dropout=params["dropout"],
            activation="gelu", batch_first=True,
        ) for _ in range(params["num_layers"])]
        # Existing default checkpoints use layer; custom sizes use layers.
        self.default_layout = params == DEFAULT_PARAMS["transformer"]
        if self.default_layout:
            self.layer = layers[0]
        else:
            self.layers = nn.ModuleList(layers)
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
        encoding = torch.zeros(seq_len, width)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[:width // 2])
        self.register_buffer("position_encoding", encoding.unsqueeze(0))
        self.register_buffer("causal_mask", torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1))

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[1]
        if length > self.position_encoding.shape[1]:
            raise ValueError("Transformer input exceeds configured seq_len")
        x = self.projection(x) + self.position_encoding[:, :length].to(dtype=x.dtype)
        for layer in ([self.layer] if self.default_layout else self.layers):
            x = layer(x, src_mask=self.causal_mask[:length, :length])
        return x


# 4. Branch construction and the independent two-network container.
def _build_encoder(name: str, input_dim: int, seq_len: int, params: dict):
    if name == "mlp":
        layers = []
        width = input_dim
        for hidden in params["hidden_dims"]:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        return nn.Sequential(*layers), width
    if name in {"cnn", "convmixer", "resnet", "tcn", "residual_tcn"}:
        encoder = ConvolutionEncoder(input_dim, name, params)
    elif name in {"lstm", "gru", "kalmannet", "small_gru"}:
        encoder = RecurrentEncoder(input_dim, name, params)
    else:
        encoder = TransformerEncoder(input_dim, seq_len, params)
    return encoder, encoder.output_dim


class IndependentBranch(nn.Module):
    """One force or location network; no softmax or force clipping is applied."""

    def __init__(self, name: str, input_dim: int, seq_len: int, output_dim: int,
                 params: Mapping | None = None):
        super().__init__()
        resolved = resolve_params(name, params)
        self.name = name.lower()
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.seq_len = _positive_int(seq_len, "seq_len")
        self.output_dim = _positive_int(output_dim, "output_dim")
        self.params = resolved
        self.encoder, hidden_dim = _build_encoder(self.name, input_dim, seq_len, resolved)
        self.head = nn.Linear(hidden_dim, output_dim)

    def encode_sequence(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] < 1 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Input must have shape (B,T,{self.input_dim}) with T >= 1")
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] < 1 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Input must have shape (B,T,{self.input_dim}) with T >= 1")
        encoded = self.encoder(x[:, -1:, :] if self.name == "mlp" else x)
        return self.head(encoded[:, -1, :])


def _storage_keys(module: nn.Module):
    return {(tensor.device, tensor.untyped_storage().data_ptr())
            for tensor in (*module.parameters(), *module.buffers()) if tensor.numel()}


class HRMEstimator(nn.Module):
    """Storage/inference container for force and location networks with no shared parameters."""

    def __init__(self, force_net: nn.Module, location_net: nn.Module):
        super().__init__()
        if force_net is location_net:
            raise ValueError("force_net and location_net must be independent instances")
        if _storage_keys(force_net) & _storage_keys(location_net):
            raise ValueError("force_net and location_net must not share parameters or buffers")
        self.force_net = force_net
        self.location_net = location_net

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return {"force": self.force_net(x), "location_logits": self.location_net(x)}


def build_branch(name: str, input_dim: int, seq_len: int, output_dim: int,
                 params: Mapping | None = None) -> IndependentBranch:
    return IndependentBranch(name, input_dim, seq_len, output_dim, params)


def build_estimator(force_name: str, location_name: str, input_dim: int,
                    seq_len: int, num_classes: int, force_params: Mapping | None = None,
                    location_params: Mapping | None = None, force_seed: int = 42,
                    location_seed: int = 43) -> HRMEstimator:
    """Build both networks afresh with independent CPU initialization seeds.

    Preserve the caller's CPU RNG state. This factory creates CPU modules; the
    caller selects the device and sets training/shuffle seeds separately.
    """
    branches = []
    for name, output_dim, params, seed in (
        (force_name, 3, force_params, force_seed),
        (location_name, num_classes, location_params, location_seed),
    ):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Branch seeds must be integers in [0, 2**63)")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            branches.append(build_branch(name, input_dim, seq_len, output_dim, params))
    return HRMEstimator(*branches)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters, excluding buffers such as positional encodings."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["count_parameters", "MODEL_DESCRIPTIONS", "MODEL_NAMES", "DEFAULT_PARAMS", "IndependentBranch", "HRMEstimator",
           "resolve_params", "build_branch", "build_estimator"]
