import os
import json
from PIL import Image


def get_dataset(labels_path, images_path, output_dir):
    # json文件
    annotations_data = []
    images_data = []
    # 遍历文件夹中的所有文件
    count = 1
    for filename in os.listdir(labels_path):
        if filename.endswith('.txt'):  # 确保只处理文本文件
            file_path = os.path.join(labels_path, filename)
            # 打开图片
            img_path = images_path + filename[:-4] + '.jpg'
            img = Image.open(img_path)

            height = img.height
            width = img.width

            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split()  # strip()用于去除行尾的换行符
                    if len(parts) == 5:  # 确保每行有5个部分
                        class_idx = int(parts[0])  # 类别信息
                        x, y, w, h = map(float, parts[1:])  # x, y, w, h转换为浮点数

                        x, y, w, h = x * width, y * height, w * width, h * height
                        x, y = x - w / 2, y - h / 2
                        bbox = [x, y, w, h]

                        annotations_data.append({
                            'id': count,
                            'image_id': int(filename[5:-4]),
                            'bbox': bbox,
                            'category_id': class_idx,
                            'area': w * h,
                            'file_name': filename[:-4] + '.jpg'
                        })

                        count += 1
                        print(f"Category: {class_idx}, x: {x}, y: {y}, w: {w}, h: {h}")

            # 图片的信息只添加1次
            images_data.append({
                'id': int(filename[5:-4]),
                'file_name': filename[:-4] + '.jpg',
                'height': height,
                'width': width
            })

    # 将数据写入JSON文件
    data = {
        'images': images_data,
        'annotations': annotations_data
    }
    json_path = '/'.join([output_dir + '.json'])
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == '__main__':
    labels_path = 'D:\\Code\\Paper-code\\DINO\\ExDark\\processed\\train\\labels\\'
    images_path = 'D:\\Code\\Paper-code\\DINO\\ExDark\\processed\\train\\images\\'
    output_path = "D:\\Code\\Paper-code\\DINO\\ExDark\\data\\annotations\\instances_train2017"
    get_dataset(labels_path, images_path, output_path)
    labels_path = 'D:\\Code\\Paper-code\\DINO\\ExDark\\processed\\val\\labels\\'
    images_path = 'D:\\Code\\Paper-code\\DINO\\ExDark\\processed\\val\\images\\'
    output_path = "D:\\Code\\Paper-code\\DINO\\ExDark\\data\\annotations\\instances_val2017"
    get_dataset(labels_path, images_path, output_path)
