from enum import Enum, auto
from typing import NamedTuple, Optional, Dict

import torch.nn as nn
from torch import Tensor
from fastonn.SelfONN import _scalar_or_tuple_1, SelfONN1d


class LinearPosition(Enum):
    """
    nn.Linear position in model
    """
    CausalSelfAttentionCAttn = auto()
    CausalSelfAttentionCProj = auto()
    MlpCFc = auto()
    MlpCProj = auto()
    GptLmHead = auto()


class SelfONN1dParams(NamedTuple):
    """
    SelfONN1d parameters.
    in_channels, out_channels and bias come from model
    """
    kernel_size: _scalar_or_tuple_1 = 1
    stride: _scalar_or_tuple_1 = 1
    padding: _scalar_or_tuple_1 = 0
    dilation: _scalar_or_tuple_1 = 1
    groups: int = 1
    q: int = 1
    padding_mode: str = 'zeros'
    mode: str = 'fast'
    dropout: Optional[float] = None


OnnConfig = Dict[LinearPosition, SelfONN1dParams]


class LinearAdapter(nn.Module):
    """
    Replaces nn.Linear with SelfONN1d according to configuration
    """
    config: OnnConfig = {}

    def __init__(self,
                 position: LinearPosition,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None):
        super(LinearAdapter, self).__init__()

        if position in self.config:
            self.layer = SelfONN1d(in_features, out_features, bias=bias, **self.config[position]._asdict())
        else:
            self.layer = nn.Linear(in_features, out_features, bias, device, dtype)

    def forward(self, x: Tensor):
        if isinstance(self.layer, SelfONN1d):
            x = self.transpose_last_two_dimensions(x)
            x = self.layer(x)
            return self.transpose_last_two_dimensions(x)

        return self.layer(x)

    @classmethod
    def configure(cls, value: OnnConfig):
        cls.config = value

        print(f"ONN config:")
        for pos, val in value.items():
            print(pos, val)

    @staticmethod
    def transpose_last_two_dimensions(x: Tensor):
        return x.transpose(-2, -1)


# SelfONN config for nn.Linear layers
# uncomment layer to enable and adjust parameters
config: OnnConfig = {
    # LinearPosition.CausalSelfAttentionCAttn: SelfONN1dParams(),
    # LinearPosition.CausalSelfAttentionCProj: SelfONN1dParams(),
    # LinearPosition.MlpCFc: SelfONN1dParams(),
    # LinearPosition.MlpCProj: SelfONN1dParams(),
    # LinearPosition.GptLmHead: SelfONN1dParams(),
}
LinearAdapter.configure(config)
