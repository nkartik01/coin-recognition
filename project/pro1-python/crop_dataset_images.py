# import matplotlib.pyplot as plt
import os
import sys
# from skimage.feature import hog
# from skimage import data, exposure
import cv2
import numpy as np
import imutils
from enhance import *
# from sklearn import svm
# from sklearn.metrics import classification_report, accuracy_score


folder = "C:/Users/aknar/Documents/image procesing/project/dataset/kaggle"
onlyfolders = [f for f in os.listdir(folder)]
# print("Working with {0} folders".format(len(onlyfolders)))
xp = [0, 64, 128, 192, 255]
fp = [0, 16, 128, 240, 255]
x = np.arange(256)
table = np.interp(x, xp, fp).astype('uint8')
clahe = cv2.createCLAHE(clipLimit=40)

for j in range(0, len(onlyfolders)):
    # folder_sequence.append(onlyfolders[j])
    subfolder = folder+"/" + onlyfolders[j]
    onlyfiles = [f for f in os.listdir(subfolder)]
    print("Working with {0} images".format(len(onlyfiles)))
    print("Image examples: ")

    print(onlyfiles[0])
    try:
        os.mkdir(
            "C:/Users/aknar/Documents/image procesing/project/dataset/try7/"+onlyfolders[j])
    except:
        print("err")
    for _file in onlyfiles:
        # print(_file)
        img = cv2.imread(subfolder+"/"+_file)

        img = imutils.resize(img, width=700)
        rgb_planes = cv2.split(img)
        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(
                diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        result = cv2.merge(result_planes)

        img = cv2.merge(result_norm_planes)
        # cv2.imshow("", img)
        # cv2.waitKey(0)
        # cv2.imshow("resized", img)
        # cv2.waitKey(0)

        height = len(img)
        width = len(img[0])
        original = img.copy()
        image = cv2.LUT(img, table)
        output = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (3, 3))
        # detect circles in the image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 0.4,
                                   700, )
        # ensure at least some circles were found
        if circles is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.blur(gray, (3, 3))
            # detect circles in the image
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                                       0.2, 700,)
            # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                # cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                # cv2.rectangle(output, (x - 5, y - 5),
                #   (x + 5, y + 5), (0, 128, 255), -1)
                # show the output image

                left = max(0, x-r)
                top = max(0, y-r)
                right = min(x+r, width)
                bottom = min(y+r, height)

                img_res = gray[top: bottom, left:right]
                img_res = cv2.equalizeHist(img_res)
                img_res = clahe.apply(img_res)
                # img_res = conservative_smoothing_gray(img_res, 5)
                # size = (3, 3)
                # shape = cv2.MORPH_RECT
                # kernel = cv2.getStructuringElement(shape, size)
                # img_res = cv2.erode(img_res, kernel)
                # img_res = cv2.GaussianBlur(img_res, (9, 9), 0)
                # img_res = cv2.LUT(img_res, table)
                img_res = stretch(img_res)
                cv2.imwrite("C:/Users/aknar/Documents/image procesing/project/dataset/try7/" +
                            onlyfolders[j]+"/"+_file, img_res)
                print(_file)
        # break
    # break

    # cv2.imshow("output", np.hstack([img, output]))
