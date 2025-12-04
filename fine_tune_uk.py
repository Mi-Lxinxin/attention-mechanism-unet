"""
fine_tune_uk.py

Fine-tune the Attention U-Net on UK patches saved as `.npy` files in `data/patches`.

Usage (PowerShell):
    python fine_tune_uk.py --patch_dir data/patches --pretrained unet-attention-4d.hdf5 --out uk_unet_att.h5 --epochs 50 --batch_size 4 --lr 5e-5

If `--pretrained` is a full model file (saved with `model.save()`), the script will try to load it; otherwise the script will build a fresh model and attempt to load weights.

"""

import os
import argparse
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam


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
    return X, y


# Minimal attention unet (same as in notebook)
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Activation, Multiply, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def conv_block(x, filters, kernel_size=3, activation='relu'):
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = Activation(activation)(x)
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = Activation(activation)(x)
    return x


def attention_gate(x, g, inter_channels):
    # Ensure spatial shapes match: upsample gating signal `g` to x's spatial size if needed
    from tensorflow.keras.layers import UpSampling2D
    theta_x = Conv2D(inter_channels, 1, strides=1, padding='same')(x)
    phi_g = Conv2D(inter_channels, 1, strides=1, padding='same')(g)
    # If shapes differ, upsample phi_g to match theta_x
    sx = theta_x.shape[1:3]
    sg = phi_g.shape[1:3]
    if None not in sx and None not in sg and (sx[0] != sg[0] or sx[1] != sg[1]):
        # compute integer upsampling factors
        up_y = int(sx[0] // sg[0]) if sg[0] != 0 else 1
        up_x = int(sx[1] // sg[1]) if sg[1] != 0 else 1
        if up_y > 1 or up_x > 1:
            phi_g = UpSampling2D(size=(up_y, up_x), interpolation='nearest')(phi_g)
    add_xg = Activation('relu')(Add()([theta_x, phi_g]))
    psi = Conv2D(1, 1, padding='same')(add_xg)
    psi = Activation('sigmoid')(psi)
    return Multiply()([x, psi])


def build_attention_unet(input_shape=(512,512,4), base_filters=16, lr=5e-4):
    inputs = Input(shape=input_shape)
    c1 = conv_block(inputs, base_filters)
    p1 = MaxPooling2D()(c1)
    c2 = conv_block(p1, base_filters*2)
    p2 = MaxPooling2D()(c2)
    c3 = conv_block(p2, base_filters*4)
    p3 = MaxPooling2D()(c3)
    c4 = conv_block(p3, base_filters*8)
    p4 = MaxPooling2D()(c4)
    c5 = conv_block(p4, base_filters*16)
    u4 = Conv2DTranspose(base_filters*8, 2, strides=2, padding='same')(c5)
    att4 = attention_gate(c4, c5, base_filters*8)
    m4 = concatenate([u4, att4])
    c6 = conv_block(m4, base_filters*8)
    u3 = Conv2DTranspose(base_filters*4, 2, strides=2, padding='same')(c6)
    att3 = attention_gate(c3, c6, base_filters*4)
    m3 = concatenate([u3, att3])
    c7 = conv_block(m3, base_filters*4)
    u2 = Conv2DTranspose(base_filters*2, 2, strides=2, padding='same')(c7)
    att2 = attention_gate(c2, c7, base_filters*2)
    m2 = concatenate([u2, att2])
    c8 = conv_block(m2, base_filters*2)
    u1 = Conv2DTranspose(base_filters, 2, strides=2, padding='same')(c8)
    att1 = attention_gate(c1, c8, base_filters)
    m1 = concatenate([u1, att1])
    c9 = conv_block(m1, base_filters)
    outputs = Conv2D(1, 1, activation='sigmoid')(c9)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_dir', default='data/patches')
    parser.add_argument('--pretrained', default='unet-attention-4d.hdf5')
    parser.add_argument('--out', default='uk_unet_att.h5')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()

    print('Loading patches from', args.patch_dir)
    X, y = load_patches(args.patch_dir, n=args.n)
    print('Loaded', X.shape, y.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build or load model
    model = None
    if os.path.exists(args.pretrained):
        try:
            print('Attempting to load pretrained model via tf.keras.models.load_model')
            model = tf.keras.models.load_model(args.pretrained, compile=False)
            print('Loaded full model file')
        except Exception as e:
            print('Could not load full model, building architecture and attempting to load weights. Error:', e)
            model = build_attention_unet(input_shape=X_train.shape[1:], base_filters=16, lr=args.lr)
            try:
                model.load_weights(args.pretrained)
                print('Loaded weights from pretrained file')
            except Exception as e2:
                print('Could not load weights:', e2)
    else:
        print('Pretrained model not found, building new model')
        model = build_attention_unet(input_shape=X_train.shape[1:], base_filters=16, lr=args.lr)

    # Simple sample weighting strategy to upweight patches with positive mask coverage
    coverages = y_train.reshape(y_train.shape[0], -1).mean(axis=1)
    # weight = 1 + (positive_fraction * factor)
    factor = np.median((1.0 - coverages) / (coverages + 1e-6)) if np.any(coverages>0) else 1.0
    sample_weights = 1.0 + (coverages > 0).astype(float) * factor
    # reshape sample_weights to (n_samples, 1, 1) so it broadcasts over spatial dims
    sample_weights = sample_weights.reshape((-1, 1, 1))

    cb = [
        ModelCheckpoint(args.out, monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        sample_weight=sample_weights
    )

    # Save training history
    np.save(args.out + '.history.npy', history.history)
    print('Finished training. Model saved to', args.out)


if __name__ == '__main__':
    main()
