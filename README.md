# Vertebral Segmentation

[![](https://img.shields.io/badge/language-python3-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/framework-TensorFlow-blue.svg)](https://www.tensorflow.org/)
[![](https://img.shields.io/badge/framework-Keras-blue.svg)](https://keras.io/)

The code is based on [matterport Mask RCNN](https://github.com/matterport/Mask_RCNN) (Python3, Keras and TensorFlow).

----

用六院数据做脊骨分割。

椎体位置标注文件在服务器上的位置： `/DATA/data/hyguan/liuyuan_sins/data/400例椎体位置.xlsx`

生成的json格式标签：`/DATA5_DB8/data/sqpeng/Projects/VertebralSegmentation/label.json`

voxel 在服务器上的位置：

1. `/DATA/data/hyguan/liuyuan_spine/data/spine_npy`  (249例)

2. `/DATA/data/hyguan/liuyuan_spine/data/cervical/npy`  (51例 颈椎)

3. `/DATA/data/yfli/dataset/data_01_19` (2_npy, 4_npy, 5_npy 共100例)

将所有矢状面提取出来，保存为npy文件，目录: `/DATA5_DB8/data/sqpeng/Projects/VertebralSegmentation/data`

* 错误标签

1190274, 1939444(512x2), 3101826(512x48), 3391383(512x78), 3521844c(x), 4074305

（暂时抛弃这些病人的数据，有效数据为 379 例。）

----

## Mask RCNN

 🌀 ***2018-10-30 17:28 Update***

现在的思路是：将脊骨数据处理成COCO的格式，然后试试 Mask RCNN 在医疗图像上的效果。 
 
### Training on Your Own Dataset

Start by reading this [blog post about the balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46). It covers the process starting from annotating images to training to using the results in a sample application.

In summary, to train the model on your own dataset you'll need to extend two classes:

```Config```
This class contains the default configuration. Subclass it and modify the attributes you need to change.

```Dataset```
This class provides a consistent way to work with any dataset. 
***It allows you to use new datasets for training without having to change 
the code of the model.*** It also supports loading multiple datasets at the
same time, which is useful if the objects you want to detect are not 
all available in one dataset. 

See examples in `samples/shapes/train_shapes.ipynb`, `samples/coco/coco.py`, `samples/balloon/balloon.py`, and `samples/nucleus/nucleus.py`.

例子 `samples/nucleus/nucleus.py` 和脊骨分割问题非常相似！
 
