# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from coco import make_coco_transforms
from DINO.util.box_ops import masks_to_boxes


class ExDark:
    def __init__(self, img_folder, ann_folder, ann_file, transforms=None, return_masks=True):
        with open(ann_file, 'r') as f:
            self.exdark = json.load(f)

        # 对“images”字段进行排序，以便它们与“annotations”对齐
        # i.e., in alphabetical order
        # self.exdark['images'] = sorted(self.exdark['images'], key=lambda x: x['id'])
        # sanity check

        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.ann_file = ann_file
        self.transforms = transforms
        self.return_masks = return_masks

    def __getitem__(self, idx):
        ann_info = self.exdark['annotations'][idx] if "annotations" in self.exdark else self.exdark['images'][idx]
        img_path = Path(self.img_folder) / ("2015_" + ann_info['file_name'])
        ann_path = Path(self.ann_folder) / ann_info['file_name']

        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        target = {'image_id': torch.tensor(ann_info['image_id'])}
        if self.return_masks:
            target['masks'] = masks

        # masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.tensor(ann_info['category'], dtype=torch.int64)

        target['labels'] = labels

        # target["boxes"] = masks_to_boxes(masks)

        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.exdark['images'])

    def get_height_and_width(self, idx):
        img_info = self.exdark['images'][idx]
        height = img_info['height']
        width = img_info['width']
        return height, width


def build(image_set, args):
    img_folder_root = Path(args.exdark_path)
    ann_folder_root = Path(args.exdark_path)
    assert img_folder_root.exists(), f'provided ExDark path {img_folder_root} does not exist'
    assert ann_folder_root.exists(), f'provided ExDark path {ann_folder_root} does not exist'
    mode = 'panoptic'
    PATHS = {
        "train": ("train", Path("annotations") / f'{mode}_train.json'),
        "val": ("val", Path("annotations") / f'{mode}_val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder_path = img_folder_root / img_folder
    ann_folder = ann_folder_root / f'{mode}_{img_folder}'
    ann_file = ann_folder_root / ann_file

    dataset = ExDark(img_folder_path, ann_folder, ann_file,
                     transforms=make_coco_transforms(image_set), return_masks=args.masks)

    return dataset


if __name__ == '__main__':
    pass
