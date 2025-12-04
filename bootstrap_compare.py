import numpy as np
import pandas as pd
import glob

def load_metrics(csvpath):
    df = pd.read_csv(csvpath)
    return df['iou'].values

def compute_zero_iou_from_masks(patch_dir):
    import glob, numpy as np
    masks = sorted(glob.glob(patch_dir+'/*_mask_*.npy'))
    zeros = []
    for m in masks:
        a = np.load(m)
        zeros.append(1.0 if a.sum()==0 else 0.0)
    return np.array(zeros)

def bootstrap_diff(obs, ref, iters=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(obs)
    diffs = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        diffs[i] = obs[idx].mean() - ref[idx].mean()
    return diffs

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_csv', default='metrics/uk_unet_trained.h5_metrics.csv')
    parser.add_argument('--patch_dir', default='data/patches')
    args = parser.parse_args()

    iou_obs = load_metrics(args.metrics_csv)
    iou_zero = compute_zero_iou_from_masks(args.patch_dir)

    diffs = bootstrap_diff(iou_obs, iou_zero, iters=5000)
    mean_diff = diffs.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    pval = (diffs <= 0).mean()

    out = {
        'obs_mean_iou': float(iou_obs.mean()),
        'zero_mean_iou': float(iou_zero.mean()),
        'mean_diff': float(mean_diff),
        'ci_2.5': float(ci_low),
        'ci_97.5': float(ci_high),
        'p_value': float(pval)
    }
    pd.Series(out).to_csv('metrics/bootstrap_summary.csv')
    print('Wrote metrics/bootstrap_summary.csv')
    print(out)
