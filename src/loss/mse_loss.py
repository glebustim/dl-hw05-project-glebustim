import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, reconstruction, lensed, **batch):
        recon_roi = reconstruction[..., 80:280, 100:366]
        lensed_roi = lensed[..., 80:280, 100:366]
        recon_roi = recon_roi.contiguous()
        lensed_roi = lensed_roi.contiguous()
        return {"loss": self.loss(recon_roi, lensed_roi)}
