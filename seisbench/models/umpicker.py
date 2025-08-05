from typing import Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import WaveformModel
from mamba_ssm.modules.mamba2 import Mamba2


class Down_SSD(nn.Module):
    """
    Downsampling block with Mamba2.

    Args:
        in_dim: Input feature dimension
        d_model: Model hidden dimension
        d_state: State space dimension
        kernel_size: Convolution kernel size
        expand: Expansion factor for Mamba2
        headdim: Head dimension for Mamba2
    """

    def __init__(
            self,
            in_dim: int,
            d_model: int,
            d_state: int = 16,
            kernel_size: int = 4,
            expand: int = 3,
            headdim: int = 3
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.d_model = d_model
        self.kernel_size = kernel_size

        self.proj = nn.Conv1d(
            in_channels=in_dim,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding="same",
            bias=False
        )

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=expand,
            headdim=headdim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.mamba(x)
        out = x.transpose(1, 2)

        return out


class Up_SSD(nn.Module):
    """
    Upsampling block with Mamba2.

    Args:
        in_dim: Input feature dimension
        d_model: Model hidden dimension
        out_dim: Output feature dimension
        d_state: State space dimension
        kernel_size: Convolution kernel size
        expand: Expansion factor for Mamba2
        headdim: Head dimension for Mamba2
    """

    def __init__(
            self,
            in_dim: int,
            d_model: int,
            out_dim: int,
            d_state: int = 16,
            kernel_size: int = 4,
            expand: int = 3,
            headdim: int = 3
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.d_model = d_model
        self.out_dim = out_dim
        self.kernel_size = kernel_size

        self.conv_same = nn.Conv1d(
            in_channels=in_dim,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding="same"
        )

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
        )

        self.out_proj = nn.Conv1d(
            in_channels=d_model,
            out_channels=out_dim,
            kernel_size=kernel_size,
            padding="same",
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        x = self.conv_same(x)
        x = x.transpose(1, 2)
        x = self.mamba(x)
        x = x.transpose(1, 2)

        out = self.out_proj(x)
        return out


class UMPicker(WaveformModel):
    """
    U-Net picker with Mamba2 state space models.

    A deep learning model for seismic phase picking that combines U-Net architecture
    with Mamba2 state space models for improved sequence modeling.

    Args:
        in_channels: Number of input channels (default: 3 for 3-component seismograms)
        classes: Number of output classes (default: 3 for N/P/S)
        phases: String representing phase labels (default: "NPS")
        sampling_rate: Sampling rate in Hz (default: 100)
        norm: Normalization method (default: "std")
        sigma: Gaussian smoothing parameter (default: 20)
        sample_boundaries: Sample boundary constraints (default: (None, None))
        **kwargs: Additional arguments passed to parent class
    """

    def __init__(
            self,
            in_channels: int = 3,
            classes: int = 3,
            phases: str = "NPS",
            sampling_rate: int = 100,
            norm: str = "std",
            sigma: int = 20,
            sample_boundaries: Tuple[Optional[int], Optional[int]] = (None, None),
            **kwargs: Any
    ) -> None:
        super().__init__(
            in_samples=3001,
            output_type="array",
            pred_sample=(0, 3001),
            labels=phases,
            sampling_rate=sampling_rate,
            **kwargs
        )

        self.in_channels = in_channels
        self.classes = classes
        self.depth = 5
        self.kernel_size = 15
        self.stride = 3
        self.dim_root = 8
        self.activation = torch.relu
        self.sigma = sigma
        self.sample_boundaries = sample_boundaries
        self.norm = norm

        self.in_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.dim_root,
            kernel_size=self.kernel_size,
            padding="same"
        )
        self.in_bn = nn.BatchNorm1d(num_features=self.dim_root, eps=1e-3)

        self.down_branch = nn.ModuleList()
        self.up_branch = nn.ModuleList()

        self._construct_encoder_path()
        self._construct_decoder_path()

        self.out = nn.Conv1d(
            in_channels=self.dim_root,
            out_channels=self.classes,
            kernel_size=1,
            padding="same"
        )
        self.softmax = nn.Softmax(dim=1)

    def _construct_encoder_path(self) -> None:
        """Build the encoder (downsampling) path."""
        last_dim = self.dim_root
        last_ks = self.kernel_size

        for i in range(self.depth):
            dim = (i + 1) * self.dim_root
            mamba_down = Down_SSD(
                in_dim=last_dim,
                d_model=dim,
                kernel_size=last_ks
            )

            conv_down = nn.Conv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=last_ks,
                stride=self.stride,
                bias=False
            )

            bn1 = nn.BatchNorm1d(num_features=dim, eps=1e-3)
            self.down_branch.append(nn.ModuleList([mamba_down, conv_down, bn1]))

            last_dim = dim
            last_ks -= 2

    def _construct_decoder_path(self) -> None:
        """Build the decoder (upsampling) path."""
        up_dim_root = self.depth * self.dim_root
        last_dim = up_dim_root
        last_ks = self.kernel_size - 2 * self.depth

        for i in range(self.depth):
            dim = up_dim_root - (i + 1) * self.dim_root
            if dim == 0:
                dim = 8
            conv_up = nn.ConvTranspose1d(
                in_channels=last_dim,
                out_channels=last_dim,
                kernel_size=last_ks,
                stride=self.stride,
                bias=False
            )

            bn2 = nn.BatchNorm1d(num_features=last_dim, eps=1e-3)

            mamba_up = Up_SSD(
                in_dim=2 * last_dim,
                d_model=last_dim,
                out_dim=dim,
                kernel_size=last_ks
            )

            self.up_branch.append(nn.ModuleList([conv_up, bn2, mamba_up]))
            last_dim = dim
            last_ks += 2

    def forward(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        """
        Forward pass through the U-Net picker.

        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)
            logits: If True, return raw logits; if False, return softmax probabilities

        Returns:
            Output tensor with phase predictions
        """
        x = self.activation(self.in_bn(self.in_conv(x)))

        skips = []

        # Encoder path
        for i, (mamba_down, conv_down, bn1) in enumerate(self.down_branch):
            if conv_down is not None:
                x = mamba_down(x)
                skips.append(x)
                x = self.activation(bn1(conv_down(x)))

        # Decoder path with skip connections
        for i, ((conv_up, bn2, mamba_up), skip) in enumerate(
                zip(self.up_branch, skips[::-1])
        ):
            x = self.activation(bn2(conv_up(x)))
            x = self._merge_skip(skip, x)
            x = mamba_up(x)

        x = self.out(x)
        if logits:
            return x
        else:
            return self.softmax(x)

    @staticmethod
    def _merge_skip(skip: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Merge skip connection with current feature map.

        Handles size mismatches by either cropping or padding.

        Args:
            skip: Skip connection tensor
            x: Current feature tensor

        Returns:
            Concatenated tensor along channel dimension
        """
        if x.shape[-1] >= skip.shape[-1]:
            offset = (x.shape[-1] - skip.shape[-1]) // 2
            x_resize = x[:, :, offset:offset + skip.shape[-1]]
        elif x.shape[-1] < skip.shape[-1]:
            offset = (skip.shape[-1] - x.shape[-1]) // 2
            x_resize = F.pad(x, (offset, skip.shape[-1] - x.shape[-1] - offset), "constant", 0)

        return torch.cat([skip, x_resize], dim=1)