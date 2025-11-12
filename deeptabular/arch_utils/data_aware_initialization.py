import torch
import torch.nn as nn


class ModuleWithInit(nn.Module):
    """Base class for pytorch module with data-aware initializer on first batch
    Helps to avoid nans in feature logits before being passed to sparsemax


    See Also
    --------

    https://github.com/yandex-research/rtdl-revisiting-models/tree/main/lib/node
    """

    def __init__(self):
        super().__init__()
        self._is_initialized_tensor = nn.Parameter(torch.tensor(0, dtype=torch.uint8), requires_grad=False)
        self._is_initialized_bool = None

    def initialize(self, *args, **kwargs):
        """Initialize module tensors using first batch of data."""
        raise NotImplementedError("Please implement ")

    def __call__(self, *args, **kwargs):
        if self._is_initialized_bool is None:
            self._is_initialized_bool = bool(self._is_initialized_tensor.item())
        if not self._is_initialized_bool:
            self.initialize(*args, **kwargs)
            self._is_initialized_tensor.data[...] = 1
            self._is_initialized_bool = True
        return super().__call__(*args, **kwargs)
