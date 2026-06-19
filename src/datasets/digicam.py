import os
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from src.lensless_helpers.utils import load_image
from src.lensless_helpers.preprocessor import get_dataset_object


class DigicamDataset(BaseDataset):
    def __init__(
        self, data_dir, split="train", limit=None, shuffle_index=False, instance_transforms=None, *args, **kwargs):
        self.data_dir = Path(data_dir) / split
        lensless_dir = self.data_dir / "lensless"
        mask_dir = self.data_dir / "masks"
        lensed_dir = self.data_dir / "lensed"

        index = []
        for img_path in sorted(lensless_dir.glob("*.png")):
            item = {
                "image_id": img_path.stem,
                "lensless_path": str(img_path),
                "mask_path": str(mask_dir / f"{img_path.stem}.npy"),
                "lensed_path": str(lensed_dir / f"{img_path.stem}.png") if lensed_dir.exists() else None
            }
            index.append(item)

        super().__init__(index, limit=limit, shuffle_index=shuffle_index, instance_transforms=instance_transforms, *args, **kwargs)

    def __getitem__(self, idx):
        item = self._index[idx]

        lensed_np = load_image(item["lensed_path"], return_float=True)
        lensless_np = load_image(item["lensless_path"], return_float=True)
        mask_vals = np.load(item["mask_path"])
        lensed, lensless, psf = get_dataset_object(lensed_np, lensless_np, mask_vals)

        if isinstance(lensless, np.ndarray):
            lensless = torch.from_numpy(lensless)
        lensless = lensless.float().permute(2, 0, 1)
        if isinstance(lensed, np.ndarray):
            lensed = torch.from_numpy(lensed)
        lensed = lensed.float().permute(2, 0, 1)
        if isinstance(psf, np.ndarray):
            psf = torch.from_numpy(psf)
        psf = psf.float()

        if psf.dim() == 4:
            psf = psf.squeeze(0)
        if psf.dim() == 3 and psf.shape[-1] in [1, 3]:
            psf = psf.permute(2, 0, 1)
        elif psf.dim() == 2:
            psf = psf.unsqueeze(0)
            
        instance_data = {
            "image_id": item["image_id"],
            "lensless": lensless,
            "mask": psf,
            "lensed": lensed
        }

        return self.preprocess_data(instance_data)
