import cv2
import numpy as np
import skimage.measure as measure
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte,filters
from skimage.filters import rank
from math import log2
from skimage import feature as ft
import math

def get_region(img):

    img = img.astype("uint8")
    ret, frame = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def Rectangle_degree(img_ori):
    contour=get_region(img_ori)
    area = []
    if len(contour)==0:
        Rectangle_degree=0
        aspect_ratio=0
        # compactness=0
    else:
        for i in range(len(contour)):
            area.append(cv2.contourArea(contour[i]))
        max_idx = np.argmax(area)
        points = contour[max_idx]
        measure_polygon = cv2.contourArea(points)
        # length = cv2.arcLength(points, True)
        hull = cv2.convexHull(points, clockwise=True)


        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        measure_rectangle = cv2.contourArea(box)
        if measure_rectangle == 0:
            return [0,0,len(contour)]
        Rectangle_degree = measure_polygon / measure_rectangle

        aspect_ratio = rect[1][0] / rect[1][1]
        if aspect_ratio > 1:
            aspect_ratio = 1 / aspect_ratio

    return [Rectangle_degree,aspect_ratio,len(contour)]

def GLCM(image):

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
    inds = np.digitize(image, bins)
    max_value = inds.max() + 1
    matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value,
                                      normed=False, symmetric=False)
    # GLCM properties
    def contrast_feature(matrix_coocurrence):
        return greycoprops(matrix_coocurrence, 'contrast')[0]

    def dissimilarity_feature(matrix_coocurrence):
        return greycoprops(matrix_coocurrence, 'dissimilarity')[0]

    def homogeneity_feature(matrix_coocurrence):
        return greycoprops(matrix_coocurrence, 'homogeneity')[0]

    def energy_feature(matrix_coocurrence):
        return greycoprops(matrix_coocurrence, 'energy')[0]

    def correlation_feature(matrix_coocurrence):
        return greycoprops(matrix_coocurrence, 'correlation')[0]

    def asm_feature(matrix_coocurrence):
        return greycoprops(matrix_coocurrence, 'ASM')[0]
    def entropy_feature(matrix_coocurrence):
        ent=[]
        x=matrix_coocurrence.shape[0]
        y=matrix_coocurrence.shape[1]
        for k in range(4):
            entropy=0
            for i in range(x):
                for j in range(y):
                    p = matrix_coocurrence[i][j][0][k] / (x * y)
                    if p !=0:
                        entropy -= p * log2(p)
                    if p>1:
                        print(matrix_coocurrence[i][j][0][k])
            ent.append(entropy)
        return np.array([ent])

    return np.hstack([contrast_feature(matrix_coocurrence),dissimilarity_feature(matrix_coocurrence),homogeneity_feature(matrix_coocurrence),\
        energy_feature(matrix_coocurrence),correlation_feature(matrix_coocurrence),asm_feature(matrix_coocurrence)])


def hog(img):
    if img.shape[0]<3 or img.shape[1]<3:
        return [0]*9
    features_all = ft.hog(img,orientations=9,pixels_per_cell=(img.shape[0],img.shape[1]),cells_per_block=(1,1),visualize=False, feature_vector= True)
    # features = ft.hog(img,orientations=8,pixels_per_cell=(math.floor(img.shape[0]/2),math.floor(img.shape[1]/2)),cells_per_block=(2,2),visualize=False, feature_vector= True)
    return features_all


def hog_context(img):
        if img.shape[0] < 4 or img.shape[1] < 4:
            return [0] * 45
        features_all = ft.hog(img, orientations=9, pixels_per_cell=(img.shape[0], img.shape[1]), cells_per_block=(1, 1),
                              visualize=False, feature_vector=True)
        features = ft.hog(img,orientations=9,pixels_per_cell=(math.floor(img.shape[0]/2),math.floor(img.shape[1]/2)),cells_per_block=(2,2),visualize=False, feature_vector= True)
        return np.hstack((features_all,features))

def RILD(img,k=10):
    # regional intensity level difference
    m, n = img.shape
    a = round((m + 1) / 2)
    b = round((n + 1) / 2)

    w = round(m / 4)
    h = round(n / 4)

    T = img[a - w:a + w, b - h:b + h]
    B = img.copy()
    B[a - w:a + w, b - h:b + h] = 0
    M_T = 0
    M_B = 0
    m_T = T.mean()
    m_B = np.sum(B) / (m * n - (2 * w) * (2 * h))

    # top_k = int(15*(math.sqrt(m*n/1600)))
    # print(k * (m * n / 1600))
    top_k = math.ceil(k * (m * n / 1600)) #best
    # top_k = int(0.05* 0.75*(m * n))
    # print(top_k)
    for i in range(top_k):
        M_T = M_T + np.max(T)
        indexmax1 = [np.where(T == np.max(T))[0][0], np.where(T == np.max(T))[1][0]]
        T[indexmax1[0], indexmax1[1]] = 0
        M_B = M_B + np.max(B)
        indexmax2 = [np.where(B == np.max(B))[0][0], np.where(B == np.max(B))[1][0]]
        B[indexmax2[0], indexmax2[1]] = 0

    M_T = M_T / top_k
    M_B = M_B / top_k
    RIL_T = M_T - m_T
    RIL_B = M_B - m_B

    W = RIL_T ** 2/ (RIL_B + 1e-10)
    return W


def PCM(img):
    # regional intensity level difference
    m, n = img.shape
    a = round((m + 1) / 2)
    b = round((n + 1) / 2)

    w = round(m / 4)
    h = round(n / 4)

    T = img[a - w:a + w, b - h:b + h]
    B1 = img[0: w, 0: h]
    B2 = img[0: w, h: 3*h]
    B3 = img[0: w, 3*h: n]
    B4 = img[w: 3*w, 3 * h: n]
    B5 = img[3 * w: m, 3 * h: n]
    B6 = img[3 * w: m, h: 3*h]
    B7 = img[3 * w: m, 0: h]
    B8 = img[w: 3*w, 0: h]

    mT = T.mean()
    mB1 = B1.mean()
    mB2 = B2.mean()
    mB3 = B3.mean()
    mB4 = B4.mean()
    mB5 = B5.mean()
    mB6 = B6.mean()
    mB7 = B7.mean()
    mB8 = B8.mean()

    # contrastlist = [mT-mB1, mT-mB2, mT-mB3, mT-mB4, mT-mB5, mT-mB6, mT-mB7, mT-mB8]
    contrastlist = [(mT - mB1) * (mT - mB5), (mT - mB2) * (mT - mB6), (mT - mB3) * (mT - mB7), (mT - mB4) * (mT - mB8)]
    contrast = min(contrastlist)

    return contrast