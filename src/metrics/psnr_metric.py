import torch
from torchmetrics.image import PeakSignalNoiseRatio
from src.metrics.base_metric import BaseMetric

class PSNRMetric(BaseMetric):
    def __init__(self, device="auto", data_range=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = PeakSignalNoiseRatio(data_range=data_range).to(device)

    def __call__(self, reconstruction, lensed, **kwargs):
        recon_roi = reconstruction[..., 80:280, 100:366]
        lensed_roi = lensed[..., 80:280, 100:366]
        recon_roi = recon_roi.contiguous()
        lensed_roi = lensed_roi.contiguous()
        return self.metric(recon_roi, lensed_roi).item()
