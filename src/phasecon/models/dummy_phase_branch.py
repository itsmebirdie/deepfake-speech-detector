"""Random-tensor stand-in for PhaseBranch so Member C can build/test fusion and the model before B7 lands.
Returns exactly the shapes in docs/INTERFACES.md section 3. Delete usage once B7 is merged."""
import torch
import torch.nn as nn


class DummyPhaseBranch(nn.Module):
    def __init__(self, t_d: int = 64, c_d: int = 128):
        super().__init__()
        self.t_d, self.c_d = t_d, c_d

    def forward(self, wav: torch.Tensor):
        b = wav.shape[0]
        return {
            "H_d": torch.randn(b, self.t_d, self.c_d, device=wav.device),
            "cons_per_sample": torch.rand(b, device=wav.device),
        }
