import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask
from pathlib import Path


def create_coco_json(image_folder, output_json):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_id = 1  # 从1开始
    category_name = "target"  # 类别名称，根据需要进行修改

    # 添加类别
    coco_format["categories"].append({
        "id": category_id,
        "name": category_name
    })

    annotation_id = 1
    image_id = 1

    # 遍历文件夹中的每张掩膜图片
    for mask_image_path in Path(image_folder).glob('*.png'):  # 更改为你需要的文件类型
        # 打开掩膜图像
        mask_image = Image.open(mask_image_path).convert('L')  # 转为灰度图
        mask_array = np.array(mask_image)

        # 创建 COCO 格式的图像条目
        image_info = {
            "id": image_id,
            "file_name": mask_image_path.name,
            "height": mask_array.shape[0],
            "width": mask_array.shape[1]
        }
        coco_format["images"].append(image_info)

        # 创建 COCO 格式的注释条目，提取轮廓并生成 RLE
        rle = mask.encode(np.asfortranarray(mask_array > 0))  # 生成 RLE
        rle['counts'] = rle['counts'].decode('utf-8')  # 转换为字符串

        annotation_info = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [],  # 你可以在这里添加多边形轮廓
            "area": int(mask_area := mask.area(rle)),
            "bbox": list(mask.toBbox(rle)),  # 生成边界框
            "iscrowd": 0
        }
        coco_format["annotations"].append(annotation_info)

        annotation_id += 1
        image_id += 1

        # 将 COCO 数据保存为 JSON 文件
    with open(output_json, 'w') as json_file:
        json.dump(coco_format, json_file)


if __name__ == "__main__":
    # 设置输入文件夹和输出 JSON 文件路径
    input_folder = "E:/03-博士\学术研究/05-红外弱小目标数据集/06-NUDT-SIRST(1327张)/masks"  # 替换为你的掩膜图像所在文件夹
    output_json_file = "NUDT-SIRST.json"  # 设置输出文件名

    create_coco_json(input_folder, output_json_file)