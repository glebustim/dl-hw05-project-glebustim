import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from src.metrics.base_metric import BaseMetric

class LPIPSMetric(BaseMetric):
    def __init__(self, device="auto", net_type="vgg", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = LearnedPerceptualImagePatchSimilarity(net_type=net_type).to(device)

    def __call__(self, reconstruction, lensed, **kwargs):
        recon_roi = reconstruction[..., 80:280, 100:366]
        lensed_roi = lensed[..., 80:280, 100:366]
        recon_roi = recon_roi.contiguous()
        lensed_roi = lensed_roi.contiguous()
        recon_roi = recon_roi * 2.0 - 1.0
        lensed_roi = lensed_roi * 2.0 - 1.0
        return self.metric(recon_roi, lensed_roi).item()
