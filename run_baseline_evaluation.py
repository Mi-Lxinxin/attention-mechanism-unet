"""
run_baseline_evaluation.py

Loads available models from `models/` (or repo root), loads patches from `data/patches` (or `dataset/...`), evaluates models on available patches and writes CSV metrics to `metrics/`.

Usage (PowerShell):
    python run_baseline_evaluation.py --patch_dir data/patches --models models/unet-attention-4d.hdf5 models/unet-attention-3d.hdf5

If no models are provided, the script attempts to load the three models used by the repo: `unet-attention-3d.hdf5`, `unet-attention-4d.hdf5`, `unet-attention-4d-atlantic.hdf5`.

"""

import os
import argparse
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
import tensorflow as tf


def load_patches(patch_dir, n=None):
    imgs = sorted(glob.glob(os.path.join(patch_dir, '*_img_*.npy')))
    masks = sorted(glob.glob(os.path.join(patch_dir, '*_mask_*.npy')))
    if not imgs:
        raise FileNotFoundError(f'No image patches found in {patch_dir}')
    if n is not None:
        imgs = imgs[:n]; masks = masks[:n]
    X = [np.load(p) for p in imgs]
    y = [np.load(p) for p in masks]
    X = np.stack(X).astype('float32')
    y = np.stack(y).astype('uint8')
    return X, y, imgs


def iou_score(y_true, y_pred, eps=1e-7):
    y_true = y_true.flatten()
    y_pred = (y_pred.flatten() > 0.5).astype(int)
    inter = (y_true & y_pred).sum()
    union = (y_true | y_pred).sum()
    if union == 0:
        return 1.0 if inter==0 else 0.0
    return inter / (union + eps)


def dice_score(y_true, y_pred, eps=1e-7):
    y_true = y_true.flatten()
    y_pred = (y_pred.flatten() > 0.5).astype(int)
    inter = (y_true & y_pred).sum()
    return (2 * inter) / (y_true.sum() + y_pred.sum() + eps)


def evaluate_model(model, X, y):
    preds = model.predict(X, batch_size=4)
    ious=[]; dices=[]; precs=[]; recs=[]; f1s=[]
    for i in range(len(X)):
        gt = y[i].reshape(-1)
        pr = preds[i].reshape(-1)
        iou = iou_score(gt, pr)
        dice = dice_score(gt, pr)
        p = precision_score(gt, (pr>0.5).astype(int), zero_division=0)
        r = recall_score(gt, (pr>0.5).astype(int), zero_division=0)
        f1 = 2*p*r/(p+r+1e-9) if (p+r)>0 else 0
        ious.append(iou); dices.append(dice); precs.append(p); recs.append(r); f1s.append(f1)
    return {'iou': np.array(ious), 'dice': np.array(dices), 'precision': np.array(precs), 'recall': np.array(recs), 'f1': np.array(f1s)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_dir', default='data/patches')
    parser.add_argument('--models', nargs='*', help='Paths to model files')
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()

    if args.models:
        model_files = args.models
    else:
        candidates = ['unet-attention-3d.hdf5','unet-attention-4d.hdf5','unet-attention-4d-atlantic.hdf5']
        model_files = [m for m in candidates if os.path.exists(m)]
    if not model_files:
        print('No model files found. Place model hdf5 files in repo root or pass with --models')
        return

    print('Loading patches...')
    X, y, img_paths = load_patches(args.patch_dir, n=args.n)
    print('X', X.shape, 'y', y.shape)

    os.makedirs('metrics', exist_ok=True)
    summary_rows = []
    for mfile in model_files:
        print('Loading model', mfile)
        try:
            model = tf.keras.models.load_model(mfile, compile=False)
        except Exception as e:
            print('Failed to load', mfile, 'error:', e)
            # Try to build architecture and load weights (fallback for HDF5 saved with custom objects)
            try:
                from fine_tune_uk import build_attention_unet
                # Build with input shape from data
                input_shape = X.shape[1:]
                print('Attempting fallback: build_attention_unet with input_shape', input_shape)
                model = build_attention_unet(input_shape=input_shape)
                model.load_weights(mfile)
                print('Loaded weights into rebuilt model from', mfile)
            except Exception as e2:
                print('Fallback failed for', mfile, 'error:', e2)
                continue
        res = evaluate_model(model, X, y)
        row = {
            'model': os.path.basename(mfile),
            'iou_mean': float(np.mean(res['iou'])),
            'iou_std': float(np.std(res['iou'])),
            'dice_mean': float(np.mean(res['dice'])),
            'precision_mean': float(np.mean(res['precision'])),
            'recall_mean': float(np.mean(res['recall'])),
            'f1_mean': float(np.mean(res['f1'])),
        }
        summary_rows.append(row)
        # save per-sample metrics
        df = pd.DataFrame({k: v for k,v in res.items()})
        csvname = os.path.join('metrics', os.path.basename(mfile) + '_metrics.csv')
        df.to_csv(csvname, index=False)
        print('Saved per-sample metrics to', csvname)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv('metrics/metrics_summary.csv', index=False)
        print('Wrote metrics/metrics_summary.csv')


if __name__ == '__main__':
    main()
