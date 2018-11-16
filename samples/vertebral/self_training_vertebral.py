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
    python3 vertebral.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 vertebral.py train --dataset=/path/to/balloon/dataset --weights=last

    # Apply color splash to an image
    python3 vertebral.py test --weights=/path/to/weights/file.h5 --image=<URL or path to file>

"""

import os
import sys
import time
import json
import math
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import skimage.draw
import datetime
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

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
        self.add_class("vertebrae", 1, "vertebrae")

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
                'vertebrae',
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
        if image_info["source"] != "vertebrae":
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


class VertebralDataset_self_training(utils.Dataset):
    def load_vertebral(self, dataset_dir, subset, iteration):
        """Load a subset of the vertebral dataset.
        dataset_dir: The root directory of the vertebral dataset.
        subset: What to load (train, val)
        """

        # Add classes. We have only one class to add.
        self.add_class("vertebrae", 1, "vertebrae")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # 所有病人id列表
        patients = os.listdir(dataset_dir)
        patients = [patient.split('.')[0] for patient in patients]

        MASK_PATH = '/DATA5_DB8/data/sqpeng/data/vertebrae_masks'

        # Add images
        for patient in patients:

            # 图片大小
            im = Image.open(os.path.join(dataset_dir, f'{patient}.png'))
            width, height = im.size

            self.add_image(
                'vertebrae',
                image_id=patient,  # use patient id as a unique image id
                path=os.path.join(dataset_dir, f'{patient}.png'),
                width=width,
                height=height,
                mask_path=os.path.join(MASK_PATH, f'iter_{iteration - 1}', f'{patient}.npy')
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
        if image_info["source"] != "vertebrae":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # mask = np.zeros([info['height'], info['width'], len(info['polygons'])], dtype=np.uint8)

        # for i, p in enumerate(info['polygons']):
        #     # Get indexes of pixels inside the polygon and set them to 1
        #     rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        #     mask[rr, cc, i] = 1

        # load mask
        mask = np.load(info['mask_path'])

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "vertebrae":
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, config, iteration=0):
    """Train the model."""
    if iteration == 0:
        # Training dataset.
        dataset_train = VertebralDataset()
        dataset_train.load_vertebral(args.dataset[:-4], "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = VertebralDataset()
        dataset_val.load_vertebral(args.dataset[:-4], "val")
        dataset_val.prepare()
    else:
        # Training dataset.
        dataset_train = VertebralDataset_self_training()
        dataset_train.load_vertebral(args.dataset, "train", iteration)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = VertebralDataset_self_training()
        dataset_val.load_vertebral(args.dataset, "val", iteration)
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


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def roi_filter(dataset, image_id, config, model):
    """ 过滤掉误分割的区域 """
    # image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    #     modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    image = dataset.load_image(image_id)
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)

    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                           dataset.image_reference(image_id)))

    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]

    # print(r['rois'])

    # 对每个roi区域的怀疑，编号与r['rois']相同
    suspect_dict = {i: 0 for i in range(len(r['rois']))}

    # ------------------------------------ 曲线拟合 ----------------------------------
    # 将ROI从上到下排序，每个tuple三个值： 1. 序号； 2. ROI区域坐标 y1, x1, y2, x2； 3. 中心点坐标: y_mean, x_mean
    ordered_rois = zip(range(len(r['rois'])),
                       r['rois'],
                       [((r['rois'][i][0] + r['rois'][i][2]) // 2, (r['rois'][i][1] + r['rois'][i][3]) // 2)
                        for i in range(len(r['rois']))])
    ordered_rois = list(sorted(ordered_rois, key=lambda x: x[1][0]))

    # pprint(ordered_rois)

    # 用三次曲线拟合中心点
    x = np.array([xx[2][0] for xx in ordered_rois])
    y = np.array([xx[2][1] for xx in ordered_rois])

    f1 = np.polyfit(x, y, 3)
    p1 = np.poly1d(f1)

    # y_vals = p1(x)

    # 计算每个中心点到拟合曲线的距离
    dist = [9999999] * len(ordered_rois)
    ordered_dist = []

    for i, roi in enumerate(ordered_rois):
        index, _, (x_m, y_m) = roi
        # 遍历所有点，求点到曲线的最近距离
        for xx in range(x[0], x[-1] + 1):
            dist[index] = min(dist[index], round(math.sqrt((x_m - xx) ** 2 + (y_m - p1(xx)) ** 2), 2))

        # 越远的点怀疑程度越大
        if dist[index] > 140:
            suspect_dict[index] += 3
        elif dist[index] > 80:
            suspect_dict[index] += 1
        elif dist[index] > 20:
            suspect_dict[index] -= 1
        elif dist[index] > 10:
            suspect_dict[index] -= 3
        else:
            suspect_dict[index] -= 4

        ordered_dist.append(dist[index])

    # print(u'每个中心点到拟合曲线的距离：', ordered_dist)

    # -------------------------------------- 竖直方向判断 --------------------------------
    vertical_overlap_dict = {}

    for i, roi in enumerate(ordered_rois):
        index, (_, s1, _, e1), _ = roi

        overlap_count = 0

        for j, o_roi in enumerate(ordered_rois):
            o_index, (_, s2, _, e2), _ = o_roi
            # 跳过同一个roi
            if o_index == index:
                continue

            # 判断 roi 和 o_roi 的重叠部分
            if max(s1, s2) < min(e1, e2):  # 有交集
                IoU = (e1 - s1 + e2 - s2 - (max(e1, s1, e2, s2) - min(e1, s1, e2, s2))) / (
                        max(e1, s1, e2, s2) - min(e1, s1, e2, s2))
            else:  # 无交集
                IoU = 0

            if IoU > 0.5:
                overlap_count += 1

        vertical_overlap_dict[index] = overlap_count

        if overlap_count > 4:
            suspect_dict[index] -= 3
        elif overlap_count > 2:
            suspect_dict[index] -= 2
        elif overlap_count > 1:
            suspect_dict[index] -= 1
        elif overlap_count == 0:
            suspect_dict[index] += 1

    # -------------------------------------- 水平方向判断 --------------------------------

    # 去掉 suspect dict > 0 的roi
    for k, v in suspect_dict.items():
        if v > 0:
            r['rois'][k] = np.array([0, 0, 0, 0])

    for i, roi in enumerate(ordered_rois):
        index, (s1, _, e1, _), _ = roi
        if suspect_dict[index] > 0:
            continue

        for j, o_roi in enumerate(ordered_rois):
            o_index, (s2, _, e2, _), _ = o_roi
            if suspect_dict[o_index] > 0:
                continue

            # 跳过同一个roi
            if o_index == index:
                continue

            # 判断 roi 和 o_roi 的重叠部分
            if max(s1, s2) < min(e1, e2):  # 有交集
                IoU = (e1 - s1 + e2 - s2 - (max(e1, s1, e2, s2) - min(e1, s1, e2, s2))) / (
                        max(e1, s1, e2, s2) - min(e1, s1, e2, s2))
            else:  # 无交集
                IoU = 0

            # 出现了冗余ROI
            if IoU > 0.6:
                if vertical_overlap_dict[index] < vertical_overlap_dict[o_index]:
                    r['rois'][index] = np.array([0, 0, 0, 0])
                else:
                    r['rois'][o_index] = np.array([0, 0, 0, 0])

    # -------------------------------------- 然后去除椎间盘 -------------------------------
    # 将ROI从上到下排序，每个tuple三个值： 1. 序号； 2. ROI； 3. mask； 4. mask 大小
    ordered_rois = zip(range(len(r['rois'])),
                       r['rois'],
                       [r['masks'][:, :, i] for i in range(len(r['rois']))],
                       [np.sum(r['masks'][:, :, i]) if np.any(r['rois'][i]) else 0 for i in
                        range(len(r['rois']))])
    ordered_rois = list(sorted(ordered_rois, key=lambda x: x[1][0]))

    # pprint(ordered_rois)

    # 面积列表
    region_list = [x[3] for x in ordered_rois]

    # print(region_list)

    for i, roi in enumerate(ordered_rois):
        index, box, mask, region = roi

        # 不考虑第一个和最后一个
        if i == 0 or i == len(ordered_rois) - 1 or region == 0:
            continue

        # 首先，和上一个ROI区域的IoU要大于50%
        (_, s1, _, e1) = box
        (_, s2, _, e2) = ordered_rois[i - 1][1]
        if max(s1, s2) < min(e1, e2):  # 有交集
            IoU = (e1 - s1 + e2 - s2 - (max(e1, s1, e2, s2) - min(e1, s1, e2, s2))) / (
                    max(e1, s1, e2, s2) - min(e1, s1, e2, s2))
        else:  # 无交集
            IoU = 0

        if IoU < 0.5:
            continue

        # 如果这个ROI区域小于相邻两个ROI的60%，就认为是椎间盘
        if region_list[i] < region_list[i - 1] * 0.6 and region_list[i] < region_list[i + 1] * 0.6:
            r['rois'][index] = np.array([0, 0, 0, 0])

        # 如果这个ROI区域只小于相邻某个ROI的60%，需要进一步判断
        elif region_list[i] < region_list[i - 1] * 0.6 or region_list[i] < region_list[i + 1] * 0.6:
            added_mask = mask.astype(np.int64) + ordered_rois[i - 1][2].astype(np.int64)
            intersection = np.sum(np.isin(added_mask, [2]))
            # print(intersection)
            if intersection > 160:
                r['rois'][index] = np.array([0, 0, 0, 0])

            added_mask = mask.astype(np.int64) + ordered_rois[i + 1][2].astype(np.int64)
            intersection = np.sum(np.isin(added_mask, [2]))
            # print(intersection)
            if intersection > 160:
                r['rois'][index] = np.array([0, 0, 0, 0])

    return r['rois']


def test(model, image_path=None):

    # 测试某一张照片
    # todo
    if image_path:
        print("")
        return
    # 测试整个测试集
    else:
        # test dataset
        # dataset = VertebralDataset_self_training()
        # dataset.load_vertebral(args.dataset, "val", 1)
        # dataset.prepare()

        dataset = VertebralDataset()
        dataset.load_vertebral(args.dataset, "val")
        dataset.prepare()

        for image_id in dataset.image_ids:
            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
            info = dataset.image_info[image_id]
            # if info['id'] != '2310393':
            #     continue
            print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                                   dataset.image_reference(image_id)))

            # Run object detection
            results = model.detect([image], verbose=1)

            filtered_roi = roi_filter(dataset, image_id, config, model)

            # Display results
            ax = get_ax(1)
            r = results[0]

            file_name = os.path.join('/DATA5_DB8/data/sqpeng/Projects/Mask-RCNN-Vertebral-Segmentation/iter0/seg_results_new',
                                     f'{info["id"]}.png')
            visualize.display_instances(image, filtered_roi, r['masks'], r['class_ids'],
                                        dataset.class_names, r['scores'], ax=ax,
                                        title="Predictions_{}".format(info['id']),
                                        save_path=file_name)

            ax = get_ax(rows=1, cols=2)
            file_name = os.path.join('/DATA5_DB8/data/sqpeng/Projects/Mask-RCNN-Vertebral-Segmentation/iter0/seg_results_compare',
                                     f'{info["id"]}.png')
            visualize.display_instances_2(image, r['rois'], filtered_roi, r['masks'], r['class_ids'],
                                          dataset.class_names, r['scores'], ax=ax,
                                          title="Predictions_{}".format(info['id']),
                                          save_path=file_name)

            print("Saved to ", file_name)


def save_masks(model, it, config):
    """ For self-training。 将训练得到的模型再应用于训练集，得到新一轮的训练标签(masks)，并保存下来。 """
    for subset in ['train', 'val']:
        if it == 0:
            dataset = VertebralDataset()
            dataset.load_vertebral(args.dataset[:-4], subset)
            dataset.prepare()
        else:
            dataset = VertebralDataset_self_training()
            dataset.load_vertebral(args.dataset, subset, it)
            dataset.prepare()

        FOLDER_PATH = f'/DATA5_DB8/data/sqpeng/data/vertebrae_masks/iter_{it}'
        if not os.path.exists(FOLDER_PATH):
            os.mkdir(FOLDER_PATH)

        for image_id in tqdm(dataset.image_ids):
            # image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            #     modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
            image = dataset.load_image(image_id)
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)

            info = dataset.image_info[image_id]
            print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                                   dataset.image_reference(image_id)))

            # Run object detection
            results = model.detect([image], verbose=1)

            # print('rois: ', results[0]['rois'])

            filtered_roi = roi_filter(dataset, image_id, config, model)

            r = results[0]

            count_masks = 0
            for i, roi in enumerate(filtered_roi):
                if np.any(roi):
                    count_masks += 1

            mask = np.zeros([image.shape[0], image.shape[1], count_masks], dtype=np.bool)

            index = 0
            for i, roi in enumerate(filtered_roi):
                if np.any(roi):
                    mask[:, :, index] = r['masks'][:, :, i]
                    index += 1

            file_name = os.path.join(FOLDER_PATH, f'{info["id"]}.npy')
            np.save(file_name, mask)

            print("Saved to ", file_name)


def self_training():
    """ 迭代训练 """
    for iteration in range(15):
        print(f'\n \n \n Starting iteration {iteration}! \n \n \n')

        # -------- training ---------
        print(f'\n \n \n Starting iteration {iteration} training phase! \n \n \n')

        logs_path = f'/DATA5_DB8/data/sqpeng/Projects/Mask-RCNN-Vertebral-Segmentation/logs/logs_{iteration}'

        config = VertebralConfig()
        config.display()

        # 加载模型，并初始化为coco的参数
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=logs_path)
        weights_path = '/DATA5_DB8/data/sqpeng/Projects/Mask-RCNN-Vertebral-Segmentation/mask_rcnn_coco.h5'
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

        train(model, config, iteration)

        # -------- generate masks -------
        print(f'\n \n \n Starting iteration {iteration} inference phase! \n \n \n')

        class InferenceConfig(VertebralConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()

        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=logs_path)
        weights_path = model.find_last()
        # print('weights path: ', weights_path)
        model.load_weights(weights_path, by_name=True)

        save_masks(model, iteration, config)


def self_training_test():
    """ 对比测试 self-training 的结果 """

    # test dataset
    dataset = VertebralDataset_self_training()
    dataset.load_vertebral(args.dataset, "val", 1)  #  第三个参数 iteration > 0 即可
    dataset.prepare()

    class InferenceConfig(VertebralConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # 加载两个模型
    log_path_0 = '/DATA5_DB8/data/sqpeng/Projects/Mask-RCNN-Vertebral-Segmentation/logs/logs_0'
    log_path_1 = '/DATA5_DB8/data/sqpeng/Projects/Mask-RCNN-Vertebral-Segmentation/logs/logs_3'
    # log_path_2 = '/DATA5_DB8/data/sqpeng/Projects/Mask-RCNN-Vertebral-Segmentation/logs/logs_2'

    model_0 = modellib.MaskRCNN(mode="inference", config=config,
                                model_dir=log_path_0)
    weights_path = model_0.find_last()
    model_0.load_weights(weights_path, by_name=True)

    model_1 = modellib.MaskRCNN(mode="inference", config=config,
                                model_dir=log_path_1)
    weights_path = model_1.find_last()
    model_1.load_weights(weights_path, by_name=True)

    # model_2 = modellib.MaskRCNN(mode="inference", config=config,
    #                             model_dir=log_path_2)
    # weights_path = model_2.find_last()
    # model_2.load_weights(weights_path, by_name=True)

    # models = [(model_0, 0), (model_1, 1), (model_2, 2)]
    models = [(model_0, 0), (model_1, 3)]

    # 测试图片
    for image_id in dataset.image_ids:
        # image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        #     modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        image = dataset.load_image(image_id)
        image, _, _, _, _ = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)

        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset.image_reference(image_id)))

        seg_results = []
        for each_model, it in models:
            # Run object detection
            results = each_model.detect([image], verbose=1)

            filtered_roi = roi_filter(dataset, image_id, config, each_model)

            # Display results
            r = results[0]

            seg_results.append([image, filtered_roi, r['masks'], r['class_ids'], dataset.class_names, r['scores'],
                                "Predictions_{}_iteration_{}".format(info['id'], it)])

            f_name = os.path.join('/DATA5_DB8/data/sqpeng/Projects/Mask-RCNN-Vertebral-Segmentation/seg_results_self_training',
                                  f'iteration{it}', f'{info["id"]}.png')
            visualize.display_instances(image, filtered_roi, r['masks'], r['class_ids'],
                                        dataset.class_names, r['scores'], ax=get_ax(1),
                                        title="Predictions_{}".format(info['id']),
                                        save_path=f_name)
            print("Saved to ", f_name)

        file_name = os.path.join('/DATA5_DB8/data/sqpeng/Projects/Mask-RCNN-Vertebral-Segmentation/seg_results_self_training/compare',
                                 f'{info["id"]}.png')
        visualize.display_instances_self_training(seg_results, save_path=file_name)

        print("Saved to ", file_name)

    return


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect vertebrals.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test' or 'save_masks' or 'self_training' or 'self_training_test'")
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
    # if args.command == "train":
    #     assert args.dataset, "Argument --dataset is required for training"
    # elif args.command == "test":
    #     assert args.image, \
    #         "Provide --image to apply color splash"
    assert args.dataset, "Argument --dataset is required"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Dataset: ", args.dataset[:-4])
    print("Logs: ", args.logs)

    if args.command == 'self_training':
        self_training()
    elif args.command == 'self_training_test':
        self_training_test()
    else:
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
            train(model, config)
        elif args.command == "test":
            test(model, image_path=args.image)
        elif args.command == 'save_masks':
            save_masks(model, 0, config)
        else:
            print("'{}' is not recognized. "
                  "Use 'train' or 'splash'".format(args.command))
