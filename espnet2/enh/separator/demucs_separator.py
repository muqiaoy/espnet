from collections import OrderedDict
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch_complex.tensor import ComplexTensor


from espnet.nets.pytorch_backend.transformer.encoder import (
    Encoder as TransformerEncoder,  # noqa: H301
)
from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.demucs import Demucs
from espnet2.enh.separator.abs_separator import AbsSeparator


is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")


class DemucsSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 1,
        chin: int = 1,
        chout: int = 1,
        depth: int = 5,
        kernel_size: int = 8,
        stride: int = 4,
        causal: bool = True,
        resample: int = 4,
        growth: int = 2,
        max_hidden: int = 10_000,
        normalize: bool = True,
        glu: bool = True,
        rescale: float = 0.1,
        floor: float = 1e-3,
        sample_rate: int = 16_000
    ):
        """Demucs separator.

        """
        super().__init__()

        self._num_spk = num_spk

        self.demucs = Demucs(
            chin=chin,
            chout=chout,
            hidden=64,
            depth=depth,
            kernel_size=kernel_size,
            stride=stride,
            causal=causal,
            resample=resample,
            growth=growth,
            max_hidden=max_hidden,
            normalize=normalize,
            glu=glu,
            rescale=rescale,
            floor=floor,
            sample_rate=sample_rate)

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model
                NOTE: not used in this model

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """

        # if complex spectrum
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input
        B, N = feature.shape

        # feature = feature.transpose(1, 2)  # B, N, L

        masks = self.demucs(feature)  # B, num_spk, N, L -> B, 1, N
        # masks = masks.transpose(1, 2)  # B, num_spk, L, N
        masks = masks.unbind(dim=1)  # List[B, L, N]

        masked = [input * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )

        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk

