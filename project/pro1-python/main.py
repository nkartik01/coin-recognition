import numpy as np
import argparse
import cv2
from skimage.feature import hog
import pickle
import imutils
from enhance import *
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
img = cv2.imread(args["image"])
img = imutils.resize(img, height=300)
height = len(img)
width = len(img[0])
original = img.copy()
xp = [0, 64, 128, 192, 255]
fp = [0, 16, 128, 240, 255]
x = np.arange(256)
table = np.interp(x, xp, fp).astype('uint8')
image = cv2.LUT(img, table)
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.blur(gray, (3, 3))
hog_features = []
y_train_labels = []
clahe = cv2.createCLAHE(clipLimit=40)

clf8 = joblib.load('filename3.pkl')
# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                           150, param1=40, param2=30, minRadius=50, maxRadius=140)
# ensure at least some circles were found
if circles is None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50)
    # ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5),
                      (x + 5, y + 5), (0, 128, 255), -1)
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
        # gr = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(img_res, (100, 100))
        # image = np.float32(gr)/255.0
        # fd = cv2.dct(image)
        a = []
        # for i in fd:
        #     a.extend(i)
        # # print(fd[0])
        # hog_features.append(a)
        # pc = [a]

        fd, hog_imge = hog(im, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(1, 1), visualize=True)
        pc = [fd]
        # sc = StandardScaler()
        # pc = sc.fit_transform(pc)
        y_pred = clf8.predict_proba(pc)
        print(max(y_pred[0]))
        y_pred = clf8.predict(pc)
        print(y_pred[0].split("_")[0])
        cv2.imshow(y_pred[0].split("_")[0], im)
        cv2.waitKey(0)
    # print(len(hog_features[0]))
    # x_train, x_test, y_train, y_test = train_test_split(
    #     pc, y_train_labels, test_size=1, random_state=0)
    # print(y_train)
    # print(pc)
    cv2.imshow("output", np.hstack([img, output]))
    cv2.waitKey(0)

    # import cv2
    # hog = cv2.HOGDescriptor()
    # # im = cv2.imread(sample)
    # h = hog.compute(image)
    # print(h)
else:

    print("No circles recognized")
    cv2.imshow("Output", img)
    cv2.waitKey(0)
