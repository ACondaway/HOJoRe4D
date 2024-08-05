# COLMAP Bounding Box Extraction Script

## 介绍

这个脚本使用COLMAP处理一个图像文件夹以生成点云文件，然后在图像中获取物体的二维bounding box。脚本包括以下主要步骤：

1. 运行COLMAP命令以处理图像数据集并生成点云文件。
2. 读取生成的点云文件，计算bounding box。
3. 在指定的图像上绘制bounding box并保存结果。

## 依赖

运行此脚本之前，请确保已安装以下Python库：

- `numpy`
- `trimesh`
- `opencv-python`

可以使用以下命令安装这些库：

```bash
pip install numpy trimesh opencv-python
```
## 使用方法
运行以下命令：
```bash
python get_HLoc_bbox.py path_to_image_folder path_to_output_folder path_to_sample_image path_to_output_image
```
例如：
```bash
python get_HLoc_bbox.py ./images ./colmap_output ./images/sample.jpg ./output/bbox_sample.jpg
```

