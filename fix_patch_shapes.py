import glob
import numpy as np
pairs = sorted(glob.glob('data/patches/*_img_*.npy'))
for img_path in pairs:
    mask_path = img_path.replace('_img_','_mask_')
    img = np.load(img_path)
    mask = np.load(mask_path)
    if img.shape[:2] != mask.shape[:2]:
        H,W = img.shape[:2]
        print('Fixing', img_path, 'img', img.shape, 'mask', mask.shape)
        # crop or pad mask to image size (prefer cropping)
        h0 = min(H, mask.shape[0])
        w0 = min(W, mask.shape[1])
        new_mask = np.zeros((H,W,1), dtype=mask.dtype)
        new_mask[:h0,:w0,0] = mask[:h0,:w0,0]
        np.save(mask_path, new_mask)
        print('Wrote fixed mask', mask_path, '->', new_mask.shape)
print('Done')
