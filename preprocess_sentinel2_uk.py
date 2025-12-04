"""
preprocess_sentinel2_uk.py

Standalone preprocessing script to:
- Read four Sentinel-2 band GeoTIFFs (B02,B03,B04,B08) for a tile
- Read a Hansen GFC loss raster (or similar) covering the tile
- Reproject/resample the label raster to the Sentinel grid (nearest neighbor)
- Extract 512x512 patches and save as NumPy arrays to an output directory

Usage (PowerShell):
    python preprocess_sentinel2_uk.py --b02 path/to/B02.tif --b03 path/to/B03.tif --b04 path/to/B04.tif --b08 path/to/B08.tif \
        --gfc path/to/gfc_loss.tif --out data/patches --prefix scot --patch 512 --stride 512

Notes:
- This script does not attempt to download Sentinel or GFC; it expects local files. Use AWS S3, Copernicus or GFW to download tiles.
- For large areas, subset rasters to the AOI before running this script to save memory.

"""

import os
import argparse
import numpy as np
import rioxarray as rxr
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def read_sentinel_tile(paths):
    """Read 4 bands and stack into (H,W,4) float32 array normalized 0-1.
    paths: dict with keys 'b02','b03','b04','b08'
    returns: img (H,W,4), transform (rasterio Affine), crs
    """
    bands = []
    da = None
    for key in ['b02','b03','b04','b08']:
        p = paths[key]
        if not os.path.exists(p):
            raise FileNotFoundError(f"Band not found: {p}")
        da = rxr.open_rasterio(p, masked=True)
        arr = np.array(da.squeeze())
        bands.append(arr)
    img = np.stack(bands, axis=-1).astype('float32')
    # per-tile normalization
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    return img, da.rio.transform(), da.rio.crs


def read_gfc_mask(path_gfc, target_crs, target_transform, target_shape, year_min=None, year_max=None):
    """Read GFC loss raster and resample+reproject to target grid. Returns binary mask aligned to image.
    target_shape: (H,W)
    """
    if not os.path.exists(path_gfc):
        raise FileNotFoundError(path_gfc)
    src = rasterio.open(path_gfc)
    data = src.read(1)
    # basic mask: >0 -> loss
    if year_min is not None or year_max is not None:
        mask = np.zeros_like(data, dtype=np.uint8)
        if year_min is None: year_min = 1
        if year_max is None: year_max = 9999
        mask[(data >= year_min) & (data <= year_max)] = 1
    else:
        mask = (data > 0).astype(np.uint8)

    dst = np.zeros(target_shape, dtype=np.uint8)
    reproject(
        source=mask,
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest
    )
    return dst


def extract_patches(image, mask, out_dir, prefix='uk', patch_size=512, stride=512, min_mask_coverage=0.0):
    os.makedirs(out_dir, exist_ok=True)
    H, W = image.shape[:2]
    idx = 0
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            img_patch = image[y:y+patch_size, x:x+patch_size, :]
            m_patch = mask[y:y+patch_size, x:x+patch_size]
            coverage = m_patch.mean()
            if coverage >= min_mask_coverage:
                img_path = os.path.join(out_dir, f"{prefix}_img_{idx}.npy")
                mask_path = os.path.join(out_dir, f"{prefix}_mask_{idx}.npy")
                np.save(img_path, img_patch.astype('float32'))
                np.save(mask_path, m_patch.reshape(patch_size, patch_size, 1).astype('uint8'))
                idx += 1
    print(f'Wrote {idx} patches to {out_dir}')


def main():
    parser = argparse.ArgumentParser(description='Preprocess Sentinel-2 tile and GFC mask into patches')
    parser.add_argument('--b02', required=True)
    parser.add_argument('--b03', required=True)
    parser.add_argument('--b04', required=True)
    parser.add_argument('--b08', required=True)
    parser.add_argument('--gfc', required=True)
    parser.add_argument('--out', default='data/patches')
    parser.add_argument('--prefix', default='uk')
    parser.add_argument('--patch', type=int, default=512)
    parser.add_argument('--stride', type=int, default=512)
    parser.add_argument('--min_mask_coverage', type=float, default=0.0)
    args = parser.parse_args()

    paths = {'b02': args.b02, 'b03': args.b03, 'b04': args.b04, 'b08': args.b08}
    print('Reading Sentinel bands...')
    img, transform, crs = read_sentinel_tile(paths)
    print('Reading and aligning GFC mask...')
    target_shape = img.shape[:2]
    mask = read_gfc_mask(args.gfc, target_crs=crs, target_transform=transform, target_shape=target_shape)
    print('Extracting patches...')
    extract_patches(img, mask, args.out, prefix=args.prefix, patch_size=args.patch, stride=args.stride, min_mask_coverage=args.min_mask_coverage)


if __name__ == '__main__':
    main()
