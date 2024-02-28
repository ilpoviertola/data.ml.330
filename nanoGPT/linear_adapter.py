from enum import Enum, auto
from typing import NamedTuple, Optional, Dict

import torch.nn as nn
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


class SelfONN1dConfig(NamedTuple):
    """
    SelfONN1d parameters
    """
    kernel_size: _scalar_or_tuple_1 = 1
    stride: _scalar_or_tuple_1 = 1
    padding: _scalar_or_tuple_1 = 0
    dilation: _scalar_or_tuple_1 = 1
    groups: int = 1
    bias: bool = True
    q: int = 1
    padding_mode: str = 'zeros'
    mode: str = 'fast'
    dropout: Optional[float] = None


class SelfONN1dPermuteConfig(SelfONN1dConfig):
    """
    SelfONN1dPermute parameters
    """
    pass


OnnConfig = Dict[LinearPosition, Optional[SelfONN1dConfig]]


class SelfONN1dPermute(SelfONN1d):
    """
    Fixes permutation when called
    """
    def __call__(self, x, *args, **kwargs):
        x = x.permute(1, 0)
        x = super().__call__(x, *args, **kwargs)
        return x.permute(1, 0)


class LinearAdapter:
    """
    Replaces nn.Linear with SelfONN1d or SelfONN1dPermute according to configuration
    """
    config: OnnConfig = {}

    @classmethod
    def configure(cls, value: OnnConfig):
        cls.config = value

        print(f"ONN config:")
        for pos, val in value.items():
            print(pos, val)

    @classmethod
    def get(cls, position: LinearPosition, in_features: int, out_features: int,
            bias: bool = True, device=None, dtype=None) -> nn.Module:
        if position in cls.config and cls.config[position] is not None:
            onn = SelfONN1dPermute if isinstance(cls.config[position], SelfONN1dPermuteConfig) else SelfONN1d
            return onn(in_channels=in_features, out_channels=out_features, **cls.config[position]._asdict())

        return nn.Linear(in_features, out_features, bias, device, dtype)


# SelfONN config
# key: LinearPosition
# value: SelfONN1dConfig -> SelfONN1d; SelfONN1dPermuteConfig -> SelfONN1dPermute; None -> nn.Linear
config: OnnConfig = {
    # TODO: not tested. fix permutation if needed
    # LinearPosition.CausalSelfAttentionCAttn: SelfONN1dConfig(),
    # LinearPosition.CausalSelfAttentionCProj: SelfONN1dConfig(),
    # LinearPosition.MlpCFc: SelfONN1dConfig(),
    # LinearPosition.MlpCProj: SelfONN1dConfig(),

    # tested. uncomment to enable and adjust parameters
    # LinearPosition.GptLmHead: SelfONN1dPermuteConfig(),
}
LinearAdapter.configure(config)
