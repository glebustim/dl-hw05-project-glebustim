import torch
from torchmetrics.functional import mean_squared_error
from src.metrics.base_metric import BaseMetric

class MSEMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, reconstruction, lensed, **kwargs):
        recon_roi = reconstruction[..., 80:280, 100:366]
        lensed_roi = lensed[..., 80:280, 100:366]
        recon_roi = recon_roi.contiguous()
        lensed_roi = lensed_roi.contiguous()
        return mean_squared_error(recon_roi, lensed_roi).item()
