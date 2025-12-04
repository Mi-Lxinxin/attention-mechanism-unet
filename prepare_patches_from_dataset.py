#!/usr/bin/env python3
"""
prepare_patches_from_dataset.py

Create `data/patches` from the repo `dataset/` TIFF files so evaluation scripts can run.

Usage:
    python prepare_patches_from_dataset.py --split training
    python prepare_patches_from_dataset.py --split validation
    python prepare_patches_from_dataset.py --split test

The script reads `dataset/<split>/images/*.tiff` and `dataset/<split>/masks/*.tiff`,
normalizes images to float32 [0,1], ensures masks are single-channel uint8, and
saves pairs as `data/patches/<split>_img_<idx>.npy` and `data/patches/<split>_mask_<idx>.npy`.
"""
import os
import argparse
import glob
import numpy as np
import rasterio


def read_image(path):
    with rasterio.open(path) as src:
        arr = src.read()
    # arr shape: (bands, H, W) -> transpose to (H, W, C)
    arr = np.transpose(arr, (1, 2, 0)).astype('float32')
    # normalize per-image
    mn = arr.min()
    mx = arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = arr - mn
    return arr


def read_mask(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
    arr = np.asarray(arr, dtype='uint8')
    # convert non-zero to 1
    arr = (arr > 0).astype('uint8')
    arr = np.expand_dims(arr, axis=-1)  # (H,W,1)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='training', choices=['training','validation','test'])
    parser.add_argument('--dataset_dir', default='dataset')
    parser.add_argument('--out_dir', default='data/patches')
    args = parser.parse_args()

    images_dir = os.path.join(args.dataset_dir, args.split, 'images')
    masks_dir = os.path.join(args.dataset_dir, args.split, 'masks')
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(images_dir)
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(masks_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(images_dir, '*')))
    mask_files = sorted(glob.glob(os.path.join(masks_dir, '*')))

    if len(image_files) != len(mask_files):
        print('Warning: different number of images and masks (images=%d masks=%d)' % (len(image_files), len(mask_files)))

    n = min(len(image_files), len(mask_files))
    print(f'Preparing {n} pairs from {images_dir} -> {args.out_dir}')

    for i in range(n):
        img_path = image_files[i]
        mask_path = mask_files[i]
        try:
            img = read_image(img_path)
            mask = read_mask(mask_path)
        except Exception as e:
            print('Failed to read pair', img_path, mask_path, 'error:', e)
            continue
        img_name = f"{args.split}_img_{i:04d}.npy"
        mask_name = f"{args.split}_mask_{i:04d}.npy"
        np.save(os.path.join(args.out_dir, img_name), img)
        np.save(os.path.join(args.out_dir, mask_name), mask)

    print('Done. Wrote', n, 'pairs to', args.out_dir)


if __name__ == '__main__':
    main()
