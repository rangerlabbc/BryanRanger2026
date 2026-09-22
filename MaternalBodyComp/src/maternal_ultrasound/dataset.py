# =============================================================================
# dataset.py
# PyTorch Dataset for maternal ultrasound images.
#
# Folder structure expected:
#   data/processed/
#     P002/
#       T1/
#         P002_Bicep_T1/   ← images here
#         P002_Quad_T1/
#         P002_Scap_T1/
#       T2/
#         ...
#     P005/
#       ...
#
# USAGE:
#   from maternal_ultrasound.dataset import MaternalDataset
# =============================================================================

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFilter

import torch
import cv2
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms


# =============================================================================
# Region detection helpers
# Matches folder names like P002_Bicep_T1, P002_Quad_T2, P002_Scap_T1
# =============================================================================

REGION_KEYWORDS = {
    "B": ["Bicep", "BICEP"],
    "Q": ["Quad", "QUAD"],
    "S": ["Scap", "SCAP"],
    "A": ["Abd", "ABD"],
}

IMAGE_EXTS = ('.jpeg', '.jpg', '.png')


def _folder_matches_region(folder_name: str, region_code: str) -> bool:
    """Returns True if folder_name contains any keyword for the given region code."""
    keywords = REGION_KEYWORDS.get(region_code, [])
    return any(kw in folder_name for kw in keywords)


def _load_image(img_path: Path, transform, speckle: bool, despeckle: bool,
                crop_fraction: float) -> torch.Tensor:
    """
    Load, preprocess, and transform a single image.

    Args:
        img_path       : path to image file
        transform      : torchvision transform to apply
        speckle        : apply median filter (5x5) for speckle reduction
        despeckle      : apply cv2 fastNlMeansDenoisingColored (stronger)
        crop_fraction  : keep only this fraction of the image height (top portion)

    Returns:
        Transformed image tensor (C, H, W)
    """
    image = Image.open(str(img_path)).convert("RGB")

    # # Crop bottom fraction (removes measurement overlays if < 1.0)
    # if crop_fraction < 1.0:
    #     w, h = image.size
    #     image = image.crop((0, 0, w, round(h * crop_fraction)))

    # Speckle filters (optional, off by default)
    if speckle:
        image = image.filter(ImageFilter.MedianFilter(size=5))

    if despeckle:
        image_np = np.array(image)
        denoised = cv2.fastNlMeansDenoisingColored(
            image_np, None, h=10, templateWindowSize=7, searchWindowSize=21
        )
        image = Image.fromarray(denoised)

    return transform(image)


class MaternalDataset(Dataset):
    """
    Loads maternal ultrasound images and paired FM/FFM labels.

    Each patient folder contains timepoint subfolders (T1, T2, T3),
    each of which contains region subfolders (Bicep, Quad, Scap, Abd).
    Images are .jpeg/.jpg files inside the region subfolders.

    Args:
        root_dir        : path to data/processed/
        labels_csv      : path to CSV with columns [Study_ID, FM, FFM,
                          Weight_visit, Length_visit]
        patient_ids     : list of Study_IDs (strings) to include
        region_combo    : which regions to use, e.g. "BQS", "BQ", "B"
                          B=Bicep, Q=Quad, S=Scap, A=Abd
        num_images      : max images to use per region per timepoint
        output          : "FM" or "FFM"
        crop            : crop fractions [B_frac, Q_frac, S_frac, A_frac]
        speckle         : apply median filter
        despeckle       : apply NL-means denoising
        augment         : 0 = no augmentation
                          5 = 5x (hflip, vflip, 2 rotations + original)
        transform       : torchvision transform; defaults to
                          Resize(256) + ToTensor + Normalize(0.5, 0.5)
    """

    def __init__(
        self,
        root_dir: str,
        labels_csv: str,
        patient_ids: list,
        region_combo: str = "BQS",
        num_images: int = 2,
        output: str = "FM",
        crop: list = None,
        speckle: bool = False,
        despeckle: bool = False,
        augment: int = 0,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.region_combo = region_combo.upper()
        self.num_images = num_images
        self.output = output
        self.crop = crop or [1.0, 1.0, 1.0, 1.0]  # B, Q, S, A
        self.speckle = speckle
        self.despeckle = despeckle
        self.augment = augment

        # Default transform: resize to 256, normalize to [-1, 1]
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

        # Load labels CSV
        df = pd.read_csv(labels_csv)
        df["Study_ID"] = df["Study_ID"].astype(str)
        self.labels = df.set_index("Study_ID")[
            ["FM", "FFM", "Weight_visit", "Length_visit"]
        ].to_dict(orient="index")

        # Only keep patients that exist on disk AND in the CSV
        self.patient_ids = [
            pid for pid in patient_ids
            if pid in self.labels and (self.root_dir / pid).is_dir()
        ]

        if len(self.patient_ids) == 0:
            raise ValueError(
                f"No valid patients found in {root_dir}. "
                "Check that patient folders exist and Study_IDs match the CSV."
            )

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_path = self.root_dir / patient_id
        images = self._load_patient_images(patient_path)
        label = self.labels[patient_id][self.output]
        weight = self.labels[patient_id]["Weight_visit"]
        length = self.labels[patient_id]["Length_visit"]
        return images, label, weight, length

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crop_for_region(self, region_code: str) -> float:
        """Return the crop fraction for this region code."""
        order = ["B", "Q", "S", "A"]
        idx = order.index(region_code) if region_code in order else 0
        return self.crop[idx]

    def _load_patient_images(self, patient_path: Path) -> list:
        """
        Walk timepoint → region subfolders and collect images for
        each active region, up to self.num_images per region.

        Returns a flat list of transformed image tensors.
        """
        images = []
        region_counts = {r: 0 for r in self.region_combo}

        # Walk: patient/T1/patient_Bicep_T1/image.jpeg
        for timepoint_dir in sorted(patient_path.iterdir()):
            if not timepoint_dir.is_dir():
                continue

            for region_dir in sorted(timepoint_dir.iterdir()):
                if not region_dir.is_dir():
                    continue

                # Identify which region this folder belongs to
                matched_region = None
                for region_code in self.region_combo:
                    if _folder_matches_region(region_dir.name, region_code):
                        matched_region = region_code
                        break

                if matched_region is None:
                    continue
                if region_counts[matched_region] >= self.num_images:
                    continue

                # Load images from this region folder
                for img_file in sorted(region_dir.iterdir()):
                    if img_file.suffix.lower() not in IMAGE_EXTS:
                        continue
                    if img_file.name.startswith('.'):
                        continue

                    crop_frac = self._crop_for_region(matched_region)
                    tensor = _load_image(
                        img_file, self.transform,
                        self.speckle, self.despeckle, crop_frac
                    )

                    # Apply augmentation if requested
                    augmented = self._augment(tensor)
                    images.extend(augmented)

                    region_counts[matched_region] += 1
                    if region_counts[matched_region] >= self.num_images:
                        break

        if len(images) == 0:
            raise ValueError(
                f"No images found for patient {patient_path.name}. "
                "Check folder structure and region_combo setting."
            )

        return images

    def _augment(self, image: torch.Tensor) -> list:
        """
        Return a list of augmented versions of the image.
        augment=0: return [original]
        augment=5: return original + hflip + vflip + 2 rotations
        """
        result = [image]

        if self.augment >= 5:
            imx = TF.hflip(image)
            result.append(imx)
            imx = TF.vflip(imx)
            result.append(imx)
            angle = random.choice([-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90])
            result.append(TF.rotate(imx, angle))
            angle = random.choice([-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90])
            result.append(TF.rotate(imx, angle))

        if self.augment >= 8:
            imx = TF.vflip(result[-1])
            result.append(imx)
            result.append(TF.rotate(imx, 15))
            result.append(TF.rotate(imx, 45))

        return result
