"""
Mask R-CNN
Configurations and data loading code for Vertebral.

Copyright (c) 2018 Shanghai Jiao Tong University.
Licensed under the MIT License (see LICENSE for details)
Written by Shiqi Peng

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import time
import json
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import skimage.draw
import datetime
from PIL import Image
from matplotlib import pyplot as plt

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.visualize import display_images

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class VertebralConfig(Config):
    """Configuration for training on Vertebral.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "vertebral"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + vertebral

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class VertebralDataset(utils.Dataset):
    def load_vertebral(self, dataset_dir, subset):
        """Load a subset of the vertebral dataset.
        dataset_dir: The root directory of the vertebral dataset.
        subset: What to load (train, val)
        """

        # Add classes. We have only one class to add.
        self.add_class("vertebral", 1, "vertebral")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # 所有病人id列表
        patients = os.listdir(dataset_dir)
        patients = [patient.split('.')[0] for patient in patients]

        # 原始label
        with open('/DATA5_DB8/data/sqpeng/Projects/VertebralSegmentation/label.json', 'r') as F:
            d = F.readlines()
        label = json.loads(d[0].strip())

        # Add images
        for patient in patients:
            patient_data = label[patient]

            # 图片大小
            im = Image.open(os.path.join(dataset_dir, f'{patient}.png'))
            width, height = im.size

            # bounding boxes
            polygons = []
            for _, v in patient_data['bbox'].items():
                # 如果四个点都有坐标
                if v['0']['y'] is not None and v['0']['z'] is not None and \
                        v['1']['y'] is not None and v['1']['z'] is not None and \
                        v['2']['y'] is not None and v['2']['z'] is not None and \
                        v['3']['y'] is not None and v['3']['z'] is not None:
                    polygons.append({
                        'all_points_x': [v['0']['y'], v['1']['y'], v['3']['y'], v['2']['y'], v['0']['y']],
                        'all_points_y': [v['0']['z'], v['1']['z'], v['3']['z'], v['2']['z'], v['0']['z']]
                    })

            self.add_image(
                'vertebral',
                image_id=patient,  # use patient id as a unique image id
                path=os.path.join(dataset_dir, f'{patient}.png'),
                width=width,
                height=height,
                polygons=polygons
            )

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a vertebral image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "vertebral":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info['height'], info['width'], len(info['polygons'])], dtype=np.uint8)

        for i, p in enumerate(info['polygons']):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "vertebral":
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = VertebralDataset()
    dataset_train.load_vertebral(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = VertebralDataset()
    dataset_val.load_vertebral(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def test(model, image_path=None):
    def get_ax(rows=1, cols=1, size=16):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.

        Adjust the size attribute to control how big to render images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
        return ax

    # 测试某一张照片
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)

        print("Saved to ", file_name)
    # 测试整个测试集
    else:
        # test dataset
        dataset = VertebralDataset()
        dataset.load_vertebral(args.dataset, "val")
        dataset.prepare()

        for image_id in dataset.image_ids:
            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
            info = dataset.image_info[image_id]
            print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                                   dataset.image_reference(image_id)))

            # Run object detection
            results = model.detect([image], verbose=1)

            # Display results
            ax = get_ax(1)
            r = results[0]
            file_name = os.path.join('/DATA5_DB8/data/sqpeng/Projects/Mask-RCNN-Vertebral-Segmentation/seg_results',
                                     f'{info["id"]}.png')
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        dataset.class_names, r['scores'], ax=ax,
                                        title="Predictions",
                                        save_path=file_name)
            print("Saved to ", file_name)

############################################################
#  COCO Evaluation
############################################################


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect vertebrals.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/vertebral/dataset/",
                        help='Directory of the Vertebral dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path to image",
                        help='Image to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    # elif args.command == "test":
    #     assert args.image, \
    #         "Provide --image to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = VertebralConfig()
    else:
        class InferenceConfig(VertebralConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

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
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "test":
        test(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
