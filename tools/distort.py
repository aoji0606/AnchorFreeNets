import cv2 as cv
import math
import numpy as np


def MyDistort(img):
    k1, k2, k3 = -0.0324797, -0.0124873, 0.00710845  # 径向畸变
    p1, p2 = 0, 0  # 切向畸变

    my_distort = np.zeros_like(img)
    for x_index in range(width):
        for y_index in range(height):
            # 归一化坐标
            x = (x_index - cx) / fx
            y = (y_index - cy) / fy
            r = math.sqrt(x ** 2 + y ** 2)

            # 计算径向畸变和切向畸变
            x_distorted = x * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6) + \
                          2 * p1 * x * y + p2 * (r ** 2 + 2 * x ** 2)
            y_distorted = y * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6) + \
                          p1 * (r ** 2 + 2 * y ** 2) + 2 * p2 * x * y

            # 投影回素平面
            x_distorted = x_distorted * fx + cx
            y_distorted = y_distorted * fy + cy

            # 最近邻插值赋值
            if (x_distorted >= 0) and (y_distorted >= 0) and (x_distorted < width) and (y_distorted < height):
                my_distort[y_index, x_index] = img[int(y_distorted), int(x_distorted)]
            else:
                my_distort[y_index, x_index] = 0
    cv.imshow("my_distort", my_distort)


if __name__ == '__main__':
    img = cv.imread('sdj.jpg')
    height, width, channel = img.shape
    rate = 2
    height = height // rate
    width = width // rate
    img = cv.resize(img, (width, height))
    cv.imshow("img", img)

    fx, fy = 929.760 / rate, 930.612 / rate  # 焦距
    cx, cy = 963.948 / rate, 573.713 / rate  # 光心
    k1, k2, k3, k4 = -0.0324797, -0.0124873, 0.00710845, -0.00250132  # 径向畸变
    p1, p2 = 0, 0  # 切向畸变

    MyDistort(img)

    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1.0]])

    dist_coef = np.array([k1, k2, p1, p2, k3])
    normal_distort = cv.undistort(img, camera_matrix, dist_coef)
    cv.imshow("normal_distort", normal_distort)

    dist_coef = np.array([k1, k2, k3, k4])
    fisheye_distort = cv.fisheye.undistortImage(img, camera_matrix, dist_coef, Knew=camera_matrix)
    cv.imshow('fisheye_distort', fisheye_distort)

    cv.waitKey(0)
