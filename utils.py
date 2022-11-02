#!/usr/bin/env python3
"""
Utility functions for tree.py implementation of Matterport's Mask R-CNN
"""

__author__ = "Daniel Lusk"

import os
import glob
import argparse
import shutil
from patchify import patchify
from tqdm import tqdm
import tifffile as tiff


def format_mrcnn_data(rgb, label, odir):
    """Copy rgb and label data into the appropriate directory structure for use with
    Matterport's Mask R-CNN library.

    Args:
        rgb (str): Directory containing source RGB files
        label (str): Directory containing source label files
        odir (str): Output directory
    """
    image_ids = sorted([x.split(".")[0] for x in os.listdir(rgb)])
    label_ids = sorted([x.split(".")[0] for x in os.listdir(label)])

    for i in tqdm(range(len(image_ids))):
        img_id = image_ids[i]
        label_id = label_ids[i]
        img_in = os.path.join(rgb, f"{img_id}.tif")
        label_in = os.path.join(label, f"{label_id}.tif")
        img_out = os.path.join(odir, f"{img_id}/image")
        label_out = os.path.join(odir, f"{img_id}/mask")
        os.makedirs(img_out)
        os.makedirs(label_out)
        shutil.copyfile(img_in, os.path.join(img_out, f"{img_id}.tif"))
        shutil.copyfile(
            label_in,
            os.path.join(label_out, f"{label_id}.tif"),
        )


def patchify_im(im, patch_size):
    """Patchify an image

    Args:
        im (ndarray): Image
        patch_size (int): Size of single side of square patches

    Returns:
        ndarray: Array of patches from source image
    """
    if len(im.shape) == 3:
        patched = patchify(
            im,
            (patch_size, patch_size, im.shape[-1]),
            step=patch_size,
        )
    else:
        patched = patchify(
            im,
            (patch_size, patch_size),
            step=patch_size,
        )
    return patched


def patchify_rgb_label(rgb_in, label_in, rgb_out, label_out, patch_size):
    """Patchify rgb and label files

    Args:
        rgb_in (str): RGB source directory
        label_in (str): Label source directory
        rgb_out (str): RGB output directory
        label_out (str): Label output directory
        patch_size (int): Size of desired patches
    """

    rgb = sorted(glob.glob(os.path.join(rgb_in, "*.tif")))
    label = sorted(glob.glob(os.path.join(label_in, "*.tif")))

    for k in range(len(rgb)):
        rgb_name = rgb[k].split("/")[-1].split(".tif")[0]
        label_name = label[k].split("/")[-1].split(".tif")[0]
        rgb_tif = tiff.imread(rgb[k])
        label_tif = tiff.imread(label[k])

        patch_rgb = patchify_im(rgb_tif, patch_size)
        patch_label = patchify_im(label_tif, patch_size)

        for i in tqdm(range(patch_rgb.shape[0])):
            for j in range(patch_rgb.shape[1]):
                tiff.imwrite(
                    os.path.join(rgb_out, f"{rgb_name}_{i}_{j}.tif"),
                    patch_rgb[i, j, 0, :, :, :],
                )
                tiff.imwrite(
                    os.path.join(label_out, f"{label_name}_{i}_{j}.tif"),
                    patch_label[i, j, :, :],
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--patch", action="store_true", default=False)
    parser.add_argument("--mrcnn", action="store_true", default=False)

    args = parser.parse_args()

    if args.patch:
        patch_size = 256

        data_dir = "../data/watershed/"
        # Unpatchified directories
        rgb_in = os.path.join(data_dir, "rgbi/loose/")
        label_in = os.path.join(data_dir, "labels/loose")

        # Patchified directories
        rgb_out = os.path.join(data_dir, f"rgbi/loose/{patch_size}/")
        label_out = os.path.join(data_dir, f"labels/loose/{patch_size}/")

        if not os.path.exists(rgb_out):
            os.makedirs(rgb_out)

        if not os.path.exists(label_out):
            os.makedirs(label_out)

        patchify_rgb_label(rgb_in, label_in, rgb_out, label_out, patch_size)

    if args.mrcnn:
        data_dir = "../data/"
        rgb = os.path.join(data_dir, "watershed/rgbi/loose/256")
        label = os.path.join(data_dir, "watershed/labels/loose/256")
        odir = os.path.join(data_dir, "mrcnn/loose/256")
        format_mrcnn_data(rgb, label, odir)
