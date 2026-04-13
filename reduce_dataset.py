"""
Reduce dataset from 5000 images per class to 1000 images per class.
Takes the first N images and copies them to a new directory.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def reduce_dataset(
    source_root: str = "images5000",
    target_root: str = "images1000",
    n_samples: int = 1000,
):
    """Reduce dataset size by taking the first N images.

    Args:
        source_root: Source directory containing the original dataset
        target_root: Target directory for the reduced dataset
        n_samples: Number of samples to keep per class
    """

    source_path = Path(source_root)
    target_path = Path(target_root)

    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print(f"Samples per class: {n_samples}")
    print("-" * 50)

    # Process degradation types
    degradation_dir = source_path / "degradation"
    target_degradation_dir = target_path / "degradation"

    if degradation_dir.exists():
        deg_types = ["blur", "haze", "rain"]
        for deg_type in deg_types:
            source_dir = degradation_dir / deg_type
            if not source_dir.exists():
                print(f"Warning: {source_dir} does not exist, skipping...")
                continue

            target_dir = target_degradation_dir / deg_type
            target_dir.mkdir(parents=True, exist_ok=True)

            # Get all image files
            image_files = list(source_dir.glob("*"))
            image_files = [f for f in image_files if f.is_file()]

            if len(image_files) < n_samples:
                print(f"Warning: {deg_type} has only {len(image_files)} files (< {n_samples})")
                n_samples_this = len(image_files)
            else:
                n_samples_this = n_samples

            # Take first n_samples images (not random)
            selected_files = sorted(image_files)[:n_samples_this]

            # Copy files
            for src_file in tqdm(selected_files, desc=f"Copying {deg_type}"):
                shutil.copy2(src_file, target_dir / src_file.name)

            print(f"{deg_type}: {n_samples_this} / {len(image_files)} copied")

    # Process resized images
    resized_dir = source_path / "resized5000_512"
    if resized_dir.exists():
        target_resized_dir = target_path / "resized1000_512"
        target_resized_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(resized_dir.glob("*"))
        image_files = [f for f in image_files if f.is_file()]

        if len(image_files) < n_samples:
            print(f"Warning: resized has only {len(image_files)} files (< {n_samples})")
            n_samples_this = len(image_files)
        else:
            n_samples_this = n_samples

        # Take first n_samples images (not random)
        selected_files = sorted(image_files)[:n_samples_this]

        for src_file in tqdm(selected_files, desc="Copying resized"):
            shutil.copy2(src_file, target_resized_dir / src_file.name)

        print(f"resized1000_512: {n_samples_this} / {len(image_files)} copied")

    print("-" * 50)
    print(f"Done! Reduced dataset saved to: {target_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reduce dataset size")
    parser.add_argument("--source", type=str, default="images5000",
                        help="Source directory")
    parser.add_argument("--target", type=str, default="images1000",
                        help="Target directory")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of samples per class")

    args = parser.parse_args()
    reduce_dataset(args.source, args.target, args.n_samples)
