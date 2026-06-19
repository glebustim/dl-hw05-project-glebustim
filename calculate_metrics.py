import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.lensless_helpers.preprocessor import (
    convert_image_to_float,
    force_rgb,
    get_cropped_lensed,
)
from src.lensless_helpers.utils import load_image
from src.metrics import LPIPSMetric, MSEMetric, PSNRMetric, SSIMMetric


def to_tensor(image, device):
    return torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device)


def load_training_target(lensed_file, lensless_file, device):
    lensed = load_image(str(lensed_file), return_float=True)
    lensless = load_image(str(lensless_file), return_float=True)

    lensed = convert_image_to_float(force_rgb(np.asarray(lensed)))
    lensless = convert_image_to_float(force_rgb(np.asarray(lensless)))
    lensed = get_cropped_lensed(lensed, lensless)
    return to_tensor(lensed, device)


def load_reconstruction(path, device):
    reconstruction = load_image(str(path), return_float=True)
    reconstruction = convert_image_to_float(force_rgb(np.asarray(reconstruction)))
    return to_tensor(reconstruction, device)


def restore_training_canvas(reconstruction, target):
    if reconstruction.shape[-2:] == target.shape[-2:]:
        return reconstruction

    target_height, target_width = target.shape[-2:]
    reconstruction_height, reconstruction_width = reconstruction.shape[-2:]
    content_mask = target.abs().sum(dim=(0, 1)) > 0
    if content_mask.any():
        content_coords = content_mask.nonzero()
        top = content_coords[:, 0].min().item()
        bottom = content_coords[:, 0].max().item() + 1
        left = content_coords[:, 1].min().item()
        right = content_coords[:, 1].max().item() + 1
    else:
        top, bottom, left, right = 0, target_height, 0, target_width

    canvas = reconstruction.new_zeros(
        reconstruction.shape[0],
        reconstruction.shape[1],
        target_height,
        target_width,
    )
    canvas[..., top:bottom, left:right] = reconstruction
    return canvas


def calculate_metric(metric, reconstruction, lensed):
    torchmetric = getattr(metric, "metric", None)
    if torchmetric is not None:
        torchmetric.reset()
    value = metric(reconstruction, lensed)
    if torchmetric is not None:
        torchmetric.reset()
    return value


def main(lensed_dir, recon_dir, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    lensed_path = Path(lensed_dir)
    recon_path = Path(recon_dir)
    lensless_path = lensed_path.parent / "lensless"
    metrics = {
        "PSNR": PSNRMetric(device=device),
        "SSIM": SSIMMetric(device=device),
        "LPIPS": LPIPSMetric(device=device),
        "MSE": MSEMetric(),
    }
    results = {name: [] for name in metrics}
    recon_files = sorted(recon_path.glob("*.png"))

    with torch.inference_mode():
        for recon_file in tqdm(recon_files):
            lensed_file = lensed_path / recon_file.name
            lensless_file = lensless_path / recon_file.name

            lensed = load_training_target(lensed_file, lensless_file, device)
            reconstruction = restore_training_canvas(load_reconstruction(recon_file, device), lensed)

            for name, metric in metrics.items():
                results[name].append(calculate_metric(metric, reconstruction, lensed))

    for name, values in results.items():
        print(f"{name}: {sum(values) / len(values)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lensed_dir", type=str, required=True)
    parser.add_argument("--recon_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    main(args.lensed_dir, args.recon_dir, args.device)
