import os
import shutil
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

def main():
    repo_id = "bezzam/DigiCam-Mirflickr-MultiMask-10K"
    print("Downloading masks")
    repo_path = snapshot_download(
        repo_id=repo_id,
        allow_patterns="masks/*",
        repo_type="dataset"
    )
    masks_src_dir = Path(repo_path) / "masks"
    print("Loading dataset")
    ds = load_dataset(repo_id)
    output_dir = Path("DigiCam-Mirflickr-MultiMask-10K")
    for split in ["train", "test"]:
        (output_dir / split / "lensless").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "lensed").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        print(f"Processing {split} split")
        for idx, item in enumerate(tqdm(ds[split])):
            image_id = f"{split}_{idx:05d}"
            
            lensless_img = item["lensless"]
            lensless_path = output_dir / split / "lensless" / f"{image_id}.png"
            lensless_img.save(lensless_path)
            
            lensed_img = item["lensed"]
            lensed_path = output_dir / split / "lensed" / f"{image_id}.png"
            lensed_img.save(lensed_path)
            
            mask_label = item["mask_label"]
            mask_src_path = masks_src_dir / f"mask_{mask_label}.npy"
            mask_dst_path = output_dir / split / "masks" / f"{image_id}.npy"
            shutil.copy(mask_src_path, mask_dst_path)
    print("Dataset downloaded")

if __name__ == "__main__":
    main()
