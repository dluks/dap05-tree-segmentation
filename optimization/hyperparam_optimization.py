#!/usr/bin/env python3
import glob

import numpy as np
import tensorflow as tf
import tifffile as tiff
from patchify import patchify
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Input,
    MaxPool2D,
)
from tensorflow.keras.metrics import BinaryIoU, MeanIoU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm


def patch_train_label(raster, labels, img_size, channels=False, merge_channel=False):
    samp_rast = tiff.imread(raster[0])
    img_base_size = samp_rast.shape[0]
    n = len(raster)
    m = (img_base_size // img_size) ** 2

    if not channels:
        channels = samp_rast.shape[-1]

    if merge_channel:
        channels += tiff.imread(merge_channel[0]).shape[-1]

    data_train = np.zeros((n * m, img_size, img_size, channels))
    data_label = np.zeros((n * m, img_size, img_size))

    for k in range(n):
        if merge_channel:
            r = np.concatenate(
                (tiff.imread(raster[k]), tiff.imread(merge_channel[k])), axis=-1
            )
        else:
            r = tiff.imread(raster[k])[..., :channels]

        # Only read in the specified number of channels from input raster
        patches_train = patchify(
            r,
            (img_size, img_size, channels),
            step=img_size,
        )
        patches_label = patchify(
            tiff.imread(labels[k]), (img_size, img_size), step=img_size
        )
        data_train[k * m : (k + 1) * m, :, :, :] = patches_train.reshape(
            -1, img_size, img_size, channels
        )
        data_label[k * m : (k + 1) * m, :, :] = patches_label.reshape(
            -1, img_size, img_size
        )

    data_label = (data_label > 0).astype("int")
    data_label = np.expand_dims(data_label, axis=-1)
    data_train = data_train.astype("float") / 255

    print(
        f"\nData sizes:\ndata_train: {data_train.shape}\ndata_label: {data_label.shape}\n"
    )

    return data_train, data_label


# Construct the U-Net
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape):
    inputs = Input(input_shape)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    b1 = conv_block(p4, 1024)
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs, name="U-Net")
    return model


def train_unet(x_train, y_train, x_test, y_test, batch_size, epochs, eta):
    input_shape = x_train.shape[1:]

    model = build_unet(input_shape)
    batch_size = batch_size
    epochs = epochs

    model.compile(
        optimizer=Adam(learning_rate=eta),
        loss="binary_crossentropy",
        metrics=[BinaryIoU(target_class_ids=[1], threshold=0.5)],
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    return model, history


# Edit me!
data_dir = "../data/"  # Your base directory for the RGB, NIR, and labels
hand_rgb_dir = f"{data_dir}rgb/"  # Subdirectory for RGB
# hand_nir_dir = f"{data_dir}train_nir/"  # Subdirectory for NIR
hand_label_dir = f"{data_dir}label/"  # Subdirectory for labels

# Patchify hand-labeled data PLUS NIR data
patch_rgb = glob.glob(f"{hand_rgb_dir}*.tif")
# patch_nir = glob.glob(f"{hand_nir_dir}*.tif")
patch_label = glob.glob(f"{hand_label_dir}*.tif")

patch_rgb.sort()
patch_label.sort()

print("Patchifying RGB + NIR data...")
data_train, data_label = patch_train_label(patch_rgb, patch_label, 128)

print("\n✅ Done.")

# Shuffle the data
rng = np.random.default_rng(seed=42)
n = len(data_train)
k = rng.choice(n, size=n, replace=False)
data_train = data_train[k]
data_label = data_label[k]

# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(
    data_train, data_label, test_size=0.1, random_state=157
)

print(
    f"\nSizes after splitting data:\n\
x_train: {x_train.shape}\n\
y_train: {y_train.shape}\n\
x_test: {x_test.shape}\n\
y_test: {y_test.shape}"
)

print("\n✅ Done.")

# GRID SEARCH SET ONE
set_name = "batch_16_32_eta_0-01_0-02"
batch_sizes = np.array([16, 32])
etas = np.array([1e-2, 2e-2])

# GRID SEARCH SET TWO
# set_name = "batch_16_32_eta_0-05_0-1"
# batch_sizes = np.array([16, 32])
# etas = np.array([5e-2, 1e-1])

# Data structure for future grid search data storage
n_folds = 10
data = np.zeros((n_folds, batch_sizes.size, etas.size, 7), dtype=object)

# Initialize the KFold
kf = KFold(n_splits=n_folds, shuffle=True, random_state=7)

# %% Run reduced grid search
epochs = 75

for i, (itrain, itest) in enumerate(
    tqdm(
        kf.split(
            x_train,
            y_train,
        ),
        desc="K-Folds",
        position=0,
        leave=True,
        total=n_folds,
    )
):
    x_train_fold = x_train[itrain]
    y_train_fold = y_train[itrain]
    x_test_fold = x_train[itest]
    y_test_fold = y_train[itest]

    for j, batch_size in enumerate(
        tqdm(batch_sizes, desc="batch size", position=1, leave=False)
    ):
        for k, eta in enumerate(tqdm(etas, desc="ETA", position=2, leave=False)):
            # Run U-Net Here
            model, history = train_unet(
                x_train_fold,
                y_train_fold,
                x_test_fold,
                y_test_fold,
                batch_size,
                epochs,
                eta,
            )

            # Loss and accuracies from each epoch
            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            iou = list(history.history.keys())[1]
            val_iou = list(history.history.keys())[3]

            # Test the model on the preserved test data
            y_pred = model.predict(x_test)

            # Convert sigmoid probability to classification
            y_pred_thresholded = y_pred > 0.5

            # Get the IoU for the test data
            biou = BinaryIoU(target_class_ids=[0, 1], threshold=0.5)
            biou.update_state(y_pred=y_pred, y_true=y_test)
            pred_biou = biou.result().numpy()

            # only for trees
            tree_biou = BinaryIoU(target_class_ids=[1], threshold=0.5)
            tree_biou.update_state(y_pred=y_pred, y_true=y_test)
            pred_tree_biou = tree_biou.result().numpy()

            # only for non-tree pixel (background)
            bg_biou = BinaryIoU(target_class_ids=[0], threshold=0.5)
            bg_biou.update_state(y_pred=y_pred, y_true=y_test)
            pred_bg_biou = bg_biou.result().numpy()

            # Log the five stats according to their K-Fold and parameter iteration
            stats = [
                pred_biou,
                pred_tree_biou,
                pred_bg_biou,
                loss,
                val_loss,
                iou,
                val_iou,
            ]

            for s, stat in enumerate(stats):
                data[i, j, k, s] = stat

            # Save the updated array each iteration
            np.save(f"nfolds_{n_folds}_{set_name}.npy", data)
