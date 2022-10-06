"""
Mask R-CNN
Train on Berlin Trees dataset from the University of
Potsdam

Licensed under the MIT License (see LICENSE for details)
Written by Daniel Lusk

------------------------------------------------------------
TODO: Change this section
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 tree.py train --dataset=/path/to/dataset --weights=imagenet

    # Train a new model starting from specific weights file
    python3 tree.py train --dataset=/path/to/dataset --weights=/path/to/weights.h5
    
    # Train a new model with a custom train/test split and randomization seed
    python3 tree.py train --dataset=/path/to/dataset --split=0.2 --seed=420 --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 tree.py train --dataset=/path/to/dataset --weights=last

    # Generate submission file
    python3 tree.py detect --dataset=/path/to/dataset --weights=<last or /path/to/weights.h5>
"""
# Set matplotlib backend
# This has to be done before other imports that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == "__main__":
    import matplotlib

    # Agg backend runs without a display
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

import os
import sys
import glob
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import tifffile as tiff
from imgaug import augmenters as iaa

# Root directory of the project
MRCNN_DIR = os.path.abspath("../Mask_RCNN/")
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(MRCNN_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(MRCNN_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/tree/")

DEFAULT_SEED = 42
DEFAULT_SPLIT = 0.1

############################################################
#  Configurations
############################################################


class TreeConfig(Config):
    """Configuration for training on the tree segmentation dataset."""

    NAME = "tree"

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 8

    # Number of classification classes (including background)
    NUM_CLASSES = 1 + 1  # Background + tree

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more proposals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Image mean (RGB)
    MEAN_PIXEL = np.array([107.0, 105.2, 101.5])

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between tree and BG
    DETECTION_MIN_CONFIDENCE = 0

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask


class TreeInferenceConfig(TreeConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################


class TreeDataset(utils.Dataset):
    def load_tree(self, data_dir, split=DEFAULT_SPLIT, val=False, seed=DEFAULT_SEED):
        """Load a subset of the tree dataset.

        data_dir: Root directory of the dataset
        split: The ratio for the training/validation split
        val: Set to True to load the validation set instead of the
        training set
        seed: provide a random seed to generate the same train/val
        split.
        """
        # Add classes. We have one class.
        # Naming the dataset tree, and the class tree
        self.add_class("tree", 1, "tree")

        image_ids = os.listdir(data_dir)

        if not seed:
            rng = np.random.default_rng()
            seed = rng.integers(1, 999, 1)[0]

        # TODO: This feels a bit overkill--probably better to just split manually
        # with numpy.
        x_train, x_test, _, _ = train_test_split(
            image_ids, image_ids, test_size=split, random_state=seed
        )

        if val:
            image_ids = x_test
        else:
            image_ids = x_train

        # Add images
        for image_id in image_ids:
            self.add_image(
                "tree",
                image_id=image_id,
                path=os.path.join(data_dir, image_id, f"image/{image_id}.tif"),
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
         masks: A bool array of shape [height, width, instance count] with
             one mask per instance.
         class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info["path"])), "mask")

        # Read mask file from .tif image and separate classes into
        # individual boolean mask layers
        mask = tiff.imread(glob.glob(f"{mask_dir}/*.tif")[0]).astype("int")
        classes = np.unique(mask)
        masks = []
        for cl in classes:
            if cl > 0:
                m = np.zeros((mask.shape[0], mask.shape[1]))
                m[np.where(mask == cl)] = 1
                masks.append(m)

        masks = np.moveaxis(np.array(masks), 0, -1)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return masks, np.ones([masks.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "tree":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################


def train(model, dataset_dir, split=DEFAULT_SPLIT, seed=DEFAULT_SEED):
    """Train the model."""
    # Training dataset.
    dataset_train = TreeDataset()
    dataset_train.load_tree(dataset_dir, split=split, seed=seed)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TreeDataset()
    dataset_val.load_tree(dataset_dir, split=split, val=True, seed=seed)
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf(
        (0, 2),
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf(
                [iaa.Affine(rotate=90), iaa.Affine(rotate=180), iaa.Affine(rotate=270)]
            ),
            iaa.Multiply((0.8, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 5.0)),
        ],
    )

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=20,
        augmentation=augmentation,
        layers="heads",
    )

    print("Train all layers")
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=40,
        augmentation=augmentation,
        layers="all",
    )


############################################################
#  RLE Encoding
############################################################


def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################


def detect(model, dataset_dir, split=DEFAULT_SPLIT, seed=DEFAULT_SEED, val=False):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = TreeDataset()
    dataset.load_tree(dataset_dir, split=split, seed=seed, val=val)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image,
            r["rois"],
            r["masks"],
            r["class_ids"],
            dataset.class_names,
            r["scores"],
            show_bbox=False,
            show_mask=False,
            title="Predictions",
        )
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


############################################################
#  Command Line
############################################################

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Mask R-CNN for nuclei counting and segmentation"
    )
    parser.add_argument("command", metavar="<command>", help="'train' or 'detect'")
    parser.add_argument(
        "--dataset",
        required=False,
        metavar="/path/to/dataset/",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--weights",
        required=True,
        metavar="/path/to/weights.h5",
        help="Path to weights .h5 file or 'coco'",
    )
    parser.add_argument(
        "--logs",
        required=False,
        default=DEFAULT_LOGS_DIR,
        metavar="/path/to/logs/",
        help="Logs and checkpoints directory (default=logs/)",
    )
    parser.add_argument(
        "--split",
        required=False,
        default=DEFAULT_SPLIT,
        metavar="Dataset train/test split",
        help='The ratio with which to split the dataset into train and test.\
use the "seed" argument to specify seed other than the DEFAULT_SEED value.',
    )
    parser.add_argument(
        "--seed",
        required=False,
        default=DEFAULT_SEED,
        metavar="Seed for the random dataset train/test split",
        help="Overrides DEFAULT_SEED to alter the random selection of the train\
test split.",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    # elif args.command == "detect":
    #     assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.split:
        print("Split: ", args.split)
    if args.seed:
        print("Seed:", args.seed)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = TreeConfig()
    else:
        config = TreeInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(
            weights_path,
            by_name=True,
            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"],
        )
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.split, args.seed)
    elif args.command == "detect":
        detect(model, args.dataset, args.split, args.seed, val=True)
    else:
        print("'{}' is not recognized. " "Use 'train' or 'detect'".format(args.command))
