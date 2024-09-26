import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask
from scipy.ndimage import label
from pathlib import Path
import cv2


def create_coco_json(image_folder, output_json):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_id = 1  # 从1开始
    category_name = "target"  # 类别名称

    # 添加类别
    coco_format["categories"].append({
        "id": category_id,
        "name": category_name
    })

    annotation_id = 1
    image_id = 1

    # 遍历文件夹中的每张掩膜图片
    for mask_image_path in Path(image_folder).glob('*.png'):
        try:
            mask_image = Image.open(mask_image_path).convert('L')
            mask_array = np.array(mask_image)

            # 确保掩码图像是二值化的
            if np.unique(mask_array).size > 2:
                print(f"Warning: Image {mask_image_path} is not binary. Converting to binary.")
                # 转换为二值图像，这里使用固定阈值128（可以根据需要调整）
                _, mask_array = cv2.threshold(mask_array, 128, 255, cv2.THRESH_BINARY)

                # 标记连通区域
            labeled_array, num_features = label(mask_array > 0)

            # 创建 COCO 格式的图像条目
            image_info = {
                "id": image_id,
                "file_name": mask_image_path.name,
                "height": mask_array.shape[0],
                "width": mask_array.shape[1]
            }
            coco_format["images"].append(image_info)

            # 为每个连通区域创建 COCO 格式的注释条目
            for region in range(1, num_features + 1):
                region_mask = (labeled_array == region).astype(np.uint8)
                rle = mask.encode(np.asfortranarray(region_mask))
                rle['counts'] = rle['counts'].decode('utf-8')

                annotation_info = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [],  # 这里不添加多边形轮廓
                    "area": int(mask.area(rle)),
                    "bbox": list(mask.toBbox(rle)),
                    "iscrowd": 0
                }
                coco_format["annotations"].append(annotation_info)
                annotation_id += 1

            image_id += 1
        except Exception as e:
            print(f"Error processing {mask_image_path}: {e}")

            # 注意：这里需要导入cv2库来处理二值化
        # 在代码顶部添加：import cv2

    # 将 COCO 数据保存为 JSON 文件
    with open(output_json, 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)  # 添加缩进以便于阅读


if __name__ == "__main__":
    input_folder = "I:/IR-small-target Datasets/SIRST/train/masks"
    output_json_file = "I:/IR-small-target Datasets/SIRST/train/train.json"
    create_coco_json(input_folder, output_json_file)