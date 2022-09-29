#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    Conv2DTranspose,
    BatchNormalization,
    Dropout,
    Lambda,
)
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate
from matplotlib import pyplot as pl
from sklearn.model_selection import train_test_split
from patchify import patchify
from scipy.ndimage import binary_fill_holes
import tifffile as tiff
import glob
import datetime


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


# for now I am just going to use what I have in terms of watershed labels
fn_patch_train = glob.glob("watershed_labels/512/watershed_rgbi_*.tif")
fn_patch_label = glob.glob("watershed_labels/512/watershed_label_*.tif")
fn_patch_train.sort()
fn_patch_label.sort()

n = len(fn_patch_train)
img_base_size = 512
img_size = 128
m = (img_base_size // img_size) ** 2

data_train = np.zeros((n * m, img_size, img_size, 4))
data_label = np.zeros((n * m, img_size, img_size))

for k in range(n):
    patches_train = patchify(
        tiff.imread(fn_patch_train[k]), (img_size, img_size, 4), step=img_size
    )
    patches_label = patchify(
        binary_fill_holes(tiff.imread(fn_patch_label[k])),
        (img_size, img_size),
        step=img_size,
    )
    data_train[k * m : (k + 1) * m, :, :, :] = patches_train.reshape(
        -1, img_size, img_size, 4
    )
    data_label[k * m : (k + 1) * m, :, :] = patches_label.reshape(
        -1, img_size, img_size
    )

data_label = (data_label > 0).astype("int")
data_label = np.expand_dims(data_label, axis=-1)
print(data_train.max(), data_label.max())
data_train = data_train.astype("float") / 255
print(data_train.max(), data_label.max())
print(data_train.shape, data_label.shape)

x_train, x_test, y_train, y_test = train_test_split(
    data_train,
    data_label,
    test_size=0.2,
)

k = np.random.randint(0, len(x_train) - 1)
fg, ax = pl.subplots(1, 2, figsize=(11, 5), dpi=170)
ax[0].imshow(x_train[k, :, :, :3])
ax[1].imshow(y_train[k, :, :, 0], cmap="gray")
pl.show()

input_shape = x_train.shape[1:]
batch_size = 32
learning_rate = 0.05
epochs = 10

model = build_unet(input_shape)
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)]
    # metrics=[tf.keras.metrics.MeanIoU(num_classes=2)] # I do not think this works because how does it threshold the p_pred?
)

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    # callbacks=[tensorboard_callback],
)

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)

tree_iou = history.history["binary_io_u"]
val_tree_iou = history.history["val_binary_io_u"]

fg, ax = pl.subplots(1, 2, figsize=(11, 5), dpi=170)
ax[0].plot(epochs, loss, "y", label="Training loss")
ax[0].plot(epochs, val_loss, "r", label="Validation loss")
ax[0].set_title("Training and validation loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].legend()

ax[1].plot(epochs, tree_iou, "y", label="Training IoU")
ax[1].plot(epochs, val_tree_iou, "r", label="Validation IoU")
ax[1].set_title("Training and validation tree IoU")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("IoU for trees")
ax[1].legend()
pl.show()

y_pred = model.predict(x_test)
y_pred_thresholded = y_pred > 0.5

from tensorflow.keras.metrics import MeanIoU, BinaryIoU

miou = MeanIoU(num_classes=2)
miou.update_state(y_pred=y_pred_thresholded, y_true=y_test)

biou = BinaryIoU(target_class_ids=[0, 1], threshold=0.5)
biou.update_state(y_pred=y_pred, y_true=y_test)

print("Mean IoU =", miou.result().numpy())
print("Binary IoU =", biou.result().numpy())

# only for trees
biou = BinaryIoU(target_class_ids=[1], threshold=0.5)
biou.update_state(y_pred=y_pred, y_true=y_test)
print("Binary IoU for class 1 =", biou.result().numpy())

# only for non-tree pixel
biou = BinaryIoU(target_class_ids=[0], threshold=0.5)
biou.update_state(y_pred=y_pred, y_true=y_test)
print("Binary IoU for class 0 =", biou.result().numpy())


threshold = 0.5
k = np.random.randint(0, len(x_test) - 1)

test_img = x_test[k]
ground_truth = y_test[k]
prediction = model.predict(np.expand_dims(test_img, 0))[0, :, :, 0]

fg, ax = pl.subplots(1, 3, figsize=(11, 5), dpi=170)
ax[0].set_title("Testing Image")
ax[0].imshow(test_img[:, :, :3])
ax[0].imshow(prediction > 0.5, cmap="gray", alpha=0.5)
ax[1].set_title("Testing Label")
ax[1].imshow(ground_truth[:, :, 0], cmap="gray")
ax[2].set_title("Prediction on test image")
im = ax[2].imshow(prediction, vmin=0, vmax=1)
cb = fg.colorbar(im, ax=ax, location="bottom")
cb.set_label("Tree probability")
pl.show()
