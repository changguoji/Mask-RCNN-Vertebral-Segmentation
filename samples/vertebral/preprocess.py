# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import logging
import numpy as np
import skimage
from PIL import Image
from tqdm import tqdm

ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_npy_to_png(source='/DATA5_DB8/data/sqpeng/Projects/VertebralSegmentation/data',
                       dest='/DATA5_DB8/data/sqpeng/data/vertebrae'):
    """
    将 npy 格式的数据转换为 png 格式存储，并随机分为 train 和 val 两部分（3：1）。

    :param source: npy 数据路径。
    :param dest: png 数据路径。
    :return:
    """
    # 获取病人列表
    patients = os.listdir(source)
    patients = [patient.split('.')[0] for patient in patients]
    random.shuffle(patients)  # 随机打乱顺序

    # 生成 train, val, test 的病人列表
    train_patients = patients[:int(len(patients) * 3 / 4)]
    val_patients = patients[int(len(patients) * 3 / 4):]

    logger.info(f'{len(train_patients)} patients in train set!')
    logger.info(f'{len(val_patients)} patients in validation set!')

    if not os.path.exists(os.path.join(dest, 'train')):
        os.mkdir(os.path.join(dest, 'train'))
    if not os.path.exists(os.path.join(dest, 'val')):
        os.mkdir(os.path.join(dest, 'val'))

    # 生成train图片
    for patient in train_patients:
        d = np.load(f'{source}/{patient}.npy')
        im = Image.fromarray(d)
        im = im.convert('RGB')
        im.save(os.path.join(dest, 'train', f'{patient}.png'))

    # 生成val图片
    for patient in val_patients:
        d = np.load(f'{source}/{patient}.npy')
        im = Image.fromarray(d)
        im = im.convert('RGB')
        im.save(os.path.join(dest, 'val', f'{patient}.png'))


def save_resized_image():
    """ 将 train & val 图片padding and resize到1024x1024，并保存下来"""
    BASIC_PATH = '/DATA5_DB8/data/sqpeng/data'

    subsets = ['train', 'val']

    if not os.path.exists(os.path.join(BASIC_PATH, 'vertebrae1024')):
        os.mkdir(os.path.join(BASIC_PATH, 'vertebrae1024'))

    for subset in subsets:
        patients = os.listdir(os.path.join(BASIC_PATH, 'vertebrae', subset))

        save_path = os.path.join(BASIC_PATH, 'vertebrae1024', subset)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for patient in tqdm(patients):
            image = skimage.io.imread(os.path.join(BASIC_PATH, 'vertebrae', subset, patient))
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=800,
                min_scale=0,
                max_dim=1024,
                mode='square')
            skimage.io.imsave(os.path.join(save_path, patient), image)


if __name__ == '__main__':
    import fire
    fire.Fire()
