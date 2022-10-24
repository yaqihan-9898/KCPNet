import numpy as np
import lib.config.config as cfg
from lib.utils.man_craft_feature import *
from sklearn import preprocessing

def crop_and_get_hcf_feature(rois,image):
    image = image[:, :, 0]
    feature = []
    contrast=[]
    for i in range(0, rois.shape[0]):

        # print(int(rois[i][1]),int(rois[i][3]), int(rois[i][2]),int(rois[i][4]))
        if int(rois[i][1]) == int(rois[i][3]) or int(rois[i][2]) == int(rois[i][4]):
            feature.append([0] * cfg.FLAGS.hcf_num-1)
            contrast.append(0)
        else:
            crop = image[int(rois[i][2]):int(rois[i][4]), int(rois[i][1]):int(rois[i][3])]
            # print(int(rois[i][2]),int(rois[i][4]),int(rois[i][1]),int(rois[i][3]))
            glcm_feature = GLCM(crop)
            hog_feature=np.array(hog(crop))
            Rectangle_degree_feature = np.array(Rectangle_degree(crop))
            x1 = rois[i][1]
            x2 = rois[i][3]
            y1 = rois[i][2]
            y2 = rois[i][4]
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            w = (x2 - x1)
            h = (y2 - y1)
            x1 = xc - w
            x2 = xc + w
            y1 = yc - h
            y2 = yc + h
            context = image[max(int(y1),0):min(int(y2),image.shape[1]), max(int(x1),0):min(int(x2),image.shape[0])]
            hog_feature_context = np.array(hog_context(context))
            contrast.append([RILD(context,6),RILD(context,10),RILD(context,16)])
            feature_tmp = np.hstack((glcm_feature, Rectangle_degree_feature,hog_feature,hog_feature_context))
            feature.append(feature_tmp)
    feature = np.array(feature)
    contrast = np.array(contrast).reshape([-1,3])/500

    ''''''
    # 标准化
    contrast=preprocessing.scale(contrast)
    feature = np.hstack((feature, contrast))
    return feature.astype(np.float32)  # 256,25
