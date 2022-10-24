import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from urllib.request import urlopen

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    RandomCrop,
    Crop,
    Compose)
import albumentations as A
import cv2

from matplotlib import pyplot as plt
import numpy as np



BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params={'format': 'pascal_voc', 'min_area': min_area,
                                     'min_visibility': min_visibility,
                                     'label_fields': ['category_id']})

def aug_process(image,box):
    image=image[0]
    cls=box[:,-1].reshape(-1,1)
    box=box[:,:-1].reshape([-1,4])
    annotations = {'image': image, 'bboxes': box, 'category_id': [1]*box.shape[0]}
    aug_method = [
        A.RandomRotate90(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.OneOf([
        #     # A.CLAHE(clip_limit=2),
        #     A.RandomBrightness(limit=0.1),
        #     A.RandomGamma(gamma_limit=(80, 120)),
        # ], p=1),
        # A.OneOf([
        #     A.GaussNoise(20),
        # ], p=0.15),
        # A.RandomContrast(limit=0.2, p=1),
        # A.RandomBrightness(limit=0.1, p=1)
        # A.Resize(height=512, width=512, p=1),
        # A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
        # A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
    ]
    aug = get_aug(aug_method)
    augmented = aug(**annotations)

    bbox, img, cls_id = augmented['bboxes'], augmented['image'], augmented['category_id']
    box_r=[]

    for box in bbox:
        box_r.append([box[0],box[1],box[0],box[3],box[2],box[3],box[2],box[1],1])
    box_r=np.array(box_r).reshape([-1,9])
    img=np.expand_dims(img, 0)
    bbox=np.array(bbox)
    bbox=np.hstack((bbox,cls))
    return img, bbox,box_r
