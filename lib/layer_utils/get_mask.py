import numpy as np
import cv2
import tensorflow as tf
import skimage.measure as measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from lib.utils.coordinate_convert import forward_convert,back_forward_convert

def get_mask(img, boxes):
    h, w, _ = img.shape
    mask = np.zeros([h, w])
    shape = boxes.shape
    for i in range(shape[0]):
        b = boxes[i]
        b = np.reshape(b[0:-1], [4, 2])

        # for j in range(4):
        #     if b[j,0]>60000:
        #         b[j, 0]=0
        xc = (b[0][0] + b[2][0]) / 2
        yc = (b[0][1] + b[2][1]) / 2
        w = abs(b[0][0] - b[2][0]) * 2.0
        h = abs(b[0][1] - b[2][1]) * 2.0
        rbox = [[xc - 1 / 2 * w, yc - 1 / 2 * h], [xc + 1 / 2 * w, yc - 1 / 2 * h], [xc + 1 / 2 * w, yc + 1 / 2 * h],
                [xc - 1 / 2 * w, yc + 1 / 2 * h]]

        rect = np.array(rbox, np.int32).reshape([4, 2])
        cv2.fillConvexPoly(mask, rect, 1)
    # for b in boxes:
    #     b = np.reshape(b[0:-1], [4, 2])
    #     rect = np.array(b, np.int32)
    #     cv2.fillConvexPoly(mask, rect, 1)
    # mask = cv2.resize(mask, dsize=(h // 16, w // 16))
    mask = np.expand_dims(mask, axis=-1)
    # print(mask.shape)
    return np.array(mask, np.float32)

def get_mask_plus(img, boxes):
    h, w, _ = img.shape
    mask = np.zeros([h, w])
    shape = boxes.shape
    for i in range(shape[0]):
        b = boxes[i]
        b = np.reshape(b[0:-1], [4, 2])
        # for j in range(4):
        #     if b[j,0]>60000:
        #         b[j, 0]=0
        # b = b[[0,1,3,2],:]
        rect = np.array(b, np.int32)
        cv2.fillConvexPoly(mask, rect, 1)
    # for b in boxes:
    #     b = np.reshape(b[0:-1], [4, 2])
    #     rect = np.array(b, np.int32)
    #     cv2.fillConvexPoly(mask, rect, 1)
    # mask = cv2.resize(mask, dsize=(h // 16, w // 16))
    mask = np.expand_dims(mask, axis=-1)
    # print(mask.shape)
    return np.array(mask, np.float32)


def get_mask_region(img,iter):
    img=np.squeeze(img)
    max_value = np.max(img)-1
    ret, ee = cv2.threshold(img, 1+0.50*max_value, 255, cv2.THRESH_BINARY)
    labeled_img, num = measure.label(ee, neighbors=4, background=0, return_num=True)
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    # ax.imshow(ee)
    # plt.savefig('E:\hyq\\baidu_net_data\Faster-RCNN-TensorFlow-Python3-master\output\mask\\' +'mask')
    # print(num)
    # cv2.imwrite('/usr/idip/idip/hyq/code/Faster-RCNN-TensorFlow-Python3-master/output/' + 'mask' + str(iter) + '.png',
    #             ee)
    boxes = []
    for region in measure.regionprops(labeled_img):
        # if region.area < 15:
        #     continue
        # print(regionprops(labeled_img)[max_label])
        # print(region.bbox)
        minr, minc, maxr, maxc = region.bbox

        boxes.append([minr, minc, maxr, maxc])

        # print(region)
        # boxes = boxes.append([minr, minc, maxr, maxc])
        # rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
        #                           fill=False, edgecolor='red', linewidth=2)
        # ax.add_patch(rect)
    # print(boxes)
    boxes = np.array(boxes).astype(np.float32)
    # plt.savefig('E:\hyq\\baidu_net_data\Faster-RCNN-TensorFlow-Python3-master\output\mask\\' + 'mask')
    # boxes=np.array(boxes)


    # plt.savefig('E:\hyq\\baidu_net_data\Faster-RCNN-TensorFlow-Python3-master\output\mask\\' + str(iter)+'.png')
    return boxes,ee.astype(np.float32)
# def get_mask_region(img,iter):
#     img=np.squeeze(img)
#     img = img.astype("uint8")
#     # max_value = np.max(img)
#     # ret, ee = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     # ret, ee = cv2.threshold(img, 0.55 * max_value, 1, cv2.THRESH_BINARY)
#     max_value = np.max(img)-1
#     ret, ee = cv2.threshold(img, 1+0.50*max_value, 255, cv2.THRESH_BINARY)
#     labeled_img, num = measure.label(ee, neighbors=4, background=0, return_num=True)
#     # if iter %10==0:
#     cv2.imwrite('/usr/idip/idip/hyq/code/Faster-RCNN-TensorFlow-Python3-master/output/' +'mask'+str(iter)+'.png',ee)
#     # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#     # ax.imshow(ee)
#     # plt.savefig('/usr/idip/idip/hyq/code/Faster-RCNN-TensorFlow-Python3-master/output/' +'mask'+str(iter)+'.png')
#     # print(num)
#     i=0
#
#     for region in measure.regionprops(labeled_img):
#         # if region.area < 15:
#         #     continue
#             # print(regionprops(labeled_img)[max_label])
#         # print(region.bbox)
#         minr, minc, maxr, maxc = region.bbox
#         if i==0:
#             boxes=np.array([minr, minc, maxr, maxc])
#         else:
#
#             box = np.array([minr, minc, maxr, maxc])
#             # print(box)
#             boxes=np.vstack((boxes,box))
#
#         # print(region)
#         # boxes = boxes.append([minr, minc, maxr, maxc])
#         i=i+1
#         # rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
#         #                           fill=False, edgecolor='red', linewidth=2)
#         # ax.add_patch(rect)
#     # print(boxes)
#     boxes=boxes.astype(np.float32)
#     # plt.savefig('E:\hyq\\baidu_net_data\Faster-RCNN-TensorFlow-Python3-master\output\mask\\' + 'mask')
#     # boxes=np.array(boxes)
#
#
#     # plt.savefig('E:\hyq\\baidu_net_data\Faster-RCNN-TensorFlow-Python3-master\output\mask\\' + str(iter)+'.png')
#     return boxes,ee.astype(np.float32)




# if __name__ == '__main__':
#     img = cv2.imread('E:\hyq\\baidu_net_data\Faster-RCNN-TensorFlow-Python3-master\output\mask\mask.png')
#     get_mask_region(img)











