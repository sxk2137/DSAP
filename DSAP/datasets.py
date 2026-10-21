# Modified: variable names aligned with the method diagram.
import os
import cv2
import glob
import torch
import random
import numpy as np
import torch.utils.data as data

from torchvision.transforms import ToTensor
from utils.data_utils import random_crop, add_JPEG_noise, synthesize
from utils.spatial_haze import synthesize_spatial_haze


IMG_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]


def list_images(folder):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def imread_checked(path, flag=cv2.IMREAD_COLOR):
    img = cv2.imread(path, flag)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def resolve_paired_clean_path(clean_folder, hazy_name):
    stem = os.path.splitext(hazy_name)[0]
    candidates = []

    if "_" in stem:
        candidates.append(stem.split("_")[0])
        candidates.append("_".join(stem.split("_")[:-1]))
    candidates.append(stem)

    expanded = []
    for name in candidates:
        expanded.append(name)
        root, ext = os.path.splitext(name)
        if ext:
            expanded.append(root)

    for name in dict.fromkeys(expanded):
        for ext in (".jpg", ".png", ".jpeg", ".bmp"):
            path = os.path.join(clean_folder, name + ext)
            if os.path.exists(path):
                return path

    tried = ", ".join(
        os.path.join(clean_folder, name + ext)
        for name in dict.fromkeys(expanded)
        for ext in (".jpg", ".png", ".jpeg", ".bmp")
    )
    raise RuntimeError(f"No clean pair found for hazy image {hazy_name}. Tried: {tried}")


class RealHazyData(data.Dataset):
    def __init__(self, image_folder, crop_size=512):
        super().__init__()
        self.crop_size = crop_size
        self.image_folder = image_folder

        self.image_paths = list_images(image_folder)
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in: {image_folder}")

        self.image_names = [os.path.basename(p) for p in self.image_paths]

    def __getitem__(self, index):
        path = self.image_paths[index]
        I_r = imread_checked(path)
        I_r = cv2.cvtColor(I_r, cv2.COLOR_BGR2RGB)

        hazy = ToTensor()(random_crop(I_r, self.crop_size)) * 2.0 - 1.0
        return hazy.clip(-1.0, 1.0), "hazy, foggy, misty, obscure, smoggy."

    def __len__(self):
        return len(self.image_paths)


class StaticPairedData(data.Dataset):
    def __init__(self, hazy_folder, clean_folder, crop_size=448):
        super().__init__()
        self.hazy_folder = hazy_folder
        self.clean_folder = clean_folder
        self.crop_size = crop_size

        self.hazy_paths = sorted(glob.glob(os.path.join(hazy_folder, "*.png")))
        if len(self.hazy_paths) == 0:
            raise RuntimeError(f"No hazy png images found in: {hazy_folder}")

        self.hazy_names = [os.path.basename(p) for p in self.hazy_paths]

    def __len__(self):
        return len(self.hazy_names)

    def __getitem__(self, index):
        hazy_name = self.hazy_names[index]

        clean_path = resolve_paired_clean_path(self.clean_folder, hazy_name)
        hazy_path = os.path.join(self.hazy_folder, hazy_name)

        clean = imread_checked(clean_path)
        hazy = imread_checked(hazy_path)

        if clean.shape[:2] != hazy.shape[:2]:
            hazy = cv2.resize(hazy, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_CUBIC)

        stack = np.concatenate((clean, hazy), axis=2)
        stack = random_crop(stack, self.crop_size)
        clean, hazy = stack[:, :, :3], stack[:, :, 3:]

        clean = clean.astype(np.float32) / 255.0
        hazy = hazy.astype(np.float32) / 255.0

        if np.random.rand() < 0.5:
            hazy = add_JPEG_noise(hazy)

        clean = ToTensor()(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
        hazy = ToTensor()(cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB))

        clean = (clean * 2.0 - 1.0).clip(-1.0, 1.0)
        hazy = hazy.clip(0.0, 1.0)

        return clean, hazy, (
            "sharp, clear-sky, vivid, well-defined, brightly lit, unobstructed, "
            "detailed, balanced composition, neutral-toned, natural contrast, "
            "evenly exposed, modern aesthetic, minimal atmospheric distortion, "
            "realistic yet polished look, high visual clarity, refreshing openness"
        )


class HybridTrainingData(data.Dataset):
    def __init__(
        self,
        syn_folder,
        real_folder,
        crop_size=448,
        p=0.3,
        use_spatial_haze=False,
        spatial_field_folder=None,
    ):
        super().__init__()

        self.p = p
        self.syn_folder = syn_folder
        self.real_folder = real_folder
        self.crop_size = crop_size
        self.use_spatial_haze = use_spatial_haze

        self.syn_paths = sorted(glob.glob(os.path.join(syn_folder, "rgb_500", "*.jpg")))
        self.syn_names = [os.path.basename(p) for p in self.syn_paths]

        self.real_paths = list_images(real_folder)
        self.real_names = [os.path.basename(p) for p in self.real_paths]

        if len(self.syn_paths) == 0:
            raise RuntimeError(f"No synthetic clean images found in: {os.path.join(syn_folder, 'rgb_500')}")

        if len(self.real_paths) == 0:
            raise RuntimeError(f"No real hazy images found in: {real_folder}")

        self.spatial_field_folder = spatial_field_folder
        if self.spatial_field_folder is None:
            self.spatial_field_folder = os.path.join(syn_folder, "MCBM")

        self.spatial_field_paths = []
        if self.use_spatial_haze:
            self.spatial_field_paths = []
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                self.spatial_field_paths.extend(glob.glob(os.path.join(self.spatial_field_folder, ext)))
            self.spatial_field_paths = sorted(self.spatial_field_paths)

            if len(self.spatial_field_paths) == 0:
                raise RuntimeError(
                    f"use_spatial_haze=True, but no MCBM maps found in: {self.spatial_field_folder}"
                )

        print(f"[HybridTrainingData] syn images: {len(self.syn_paths)}")
        print(f"[HybridTrainingData] real images: {len(self.real_paths)}")
        print(f"[HybridTrainingData] use_spatial_haze: {self.use_spatial_haze}")
        print(f"[HybridTrainingData] MCBM maps: {len(self.spatial_field_paths)}")

    def __getitem__(self, index):
        if np.random.rand() < self.p:
            syn_clean_path = self.syn_paths[index % len(self.syn_paths)]
            syn_clean_name = os.path.basename(syn_clean_path)

            depth_path = os.path.join(
                self.syn_folder,
                "depth_500",
                os.path.splitext(syn_clean_name)[0] + ".npy",
            )

            J = imread_checked(syn_clean_path).astype(np.float32) / 255.0

            if not os.path.exists(depth_path):
                raise RuntimeError(f"Depth file not found: {depth_path}")

            depth = np.load(depth_path)

            if self.use_spatial_haze:
                m_path = random.choice(self.spatial_field_paths)
                beta_tilde = imread_checked(m_path, cv2.IMREAD_GRAYSCALE)

                I_s = synthesize_spatial_haze(
                    J=J,
                    d=depth,
                    beta_tilde=beta_tilde,
                ).astype(np.float32) / 255.0
            else:
                I_s = synthesize(
                    J=J,
                    d=depth,
                ).astype(np.float32) / 255.0

            # J uses OpenCV BGR order; synthesize_spatial_haze returns RGB.
            J = cv2.cvtColor(J, cv2.COLOR_BGR2RGB)

            if J.shape[:2] != I_s.shape[:2]:
                I_s = cv2.resize(
                    I_s,
                    (J.shape[1], J.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )

            stack = np.concatenate((J, I_s), axis=2)
            stack = random_crop(stack, self.crop_size)

            clean, hazy = stack[:, :, :3], stack[:, :, 3:]

            clean = ToTensor()(clean).clip(0.0, 1.0)
            hazy = ToTensor()(hazy).clip(0.0, 1.0)

            # Stage1:
            # target = hazy [-1, 1]
            # condition = clean [0, 1]
            return (
                hazy * 2.0 - 1.0,
                clean,
                "hazy, foggy, misty, obscure, smoggy.",
                "cond",
            )

        else:
            real_path = random.choice(self.real_paths)

            I_r = imread_checked(real_path)
            I_r = cv2.cvtColor(I_r, cv2.COLOR_BGR2RGB)

            I_r = ToTensor()(random_crop(I_r, self.crop_size)) * 2.0 - 1.0

            return (
                I_r.clip(-1.0, 1.0),
                torch.zeros_like(I_r, dtype=I_r.dtype),
                "hazy, foggy, misty, obscure, smoggy.",
                "uncond",
            )

    def __len__(self):
        return len(self.syn_paths)
