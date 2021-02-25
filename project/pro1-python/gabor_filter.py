# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 22:14:36 2018

@author: Narang
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 13:15:57 2018

@author: Narang
"""


# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
# ksize - size of gabor filter (n, n)
# sigma - standard deviation of the gaussian function
# theta - orientation of the normal to the parallel stripes
# lambda - wavelength of the sunusoidal factor
# gamma - spatial aspect ratio
# psi - phase offset
# ktype - type and range of values that each pixel in the gabor kernel can hold


""" gabor features """

#folder = "E:/characterdatabase_munish_mru_extended_final"




from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
import numpy as np
import cv2
from skimage import data, exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import imutils
import os, sys
from sklearn.metrics import classification_report, accuracy_score
def myimshow(I, **kwargs):
    # utility function to show image
    plt.figure()
    plt.axis('off')
    plt.imshow(I, cmap=plt.gray(), **kwargs)


folder = "C:/users/aknar/documents/image procesing/project/dataset/try1"
onlyfolders = [f for f in os.listdir(folder)]
print("Working with {0} folders".format(len(onlyfolders)))
folder_sequence = []


train_files = []
y_train_labels = []
complete_file_name = []

filtered_features = []
kernel_features = []
for j in range(0, len(onlyfolders)):
    folder_sequence.append(onlyfolders[j])
    subfolder = folder+"/" + onlyfolders[j]
    onlyfiles = [f for f in os.listdir(subfolder)]
    print("Working with {0} images".format(len(onlyfiles)))
    print("Image examples: ")

    print(onlyfiles[0])
    p = 1
    for _file in onlyfiles:
        # train_files.append(_file)
        # complete_file_name.append(subfolder+"/"+_file)
        label_in_file = j
        y_train_labels.append(label_in_file)

        #image = data.astronaut()
        filename = subfolder+"/"+_file

        im = cv2.imread(filename)
        """cv2.imshow('image1', im)
        plt.imshow(im, cmap = 'gray', interpolation = 'bicubic')"""

        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = imutils.resize(im, width=100)
        g_kernel = cv2.getGaborKernel(
            (3, 3), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

        filtered_img = cv2.filter2D(im, cv2.CV_8UC3, g_kernel)

        # cv2.imshow('image',img)
        #cv2.imshow('filtered image', filtered_img)
        print(filtered_img.flatten())
        filtered_features.append(filtered_img.flatten())
        # h, w = g_kernel.shape[:2]
        # g_kernel = cv2.resize(g_kernel, (3*w, 3*h),
        #                       interpolation=cv2.INTER_CUBIC)
        #cv2.imshow('gabor kernel (resized)', g_kernel)
        # kernel_features.append(g_kernel.flatten())
        # if p == 1:
        #     myimshow(filtered_img)
        #     cv2.imshow('image', img)
        #     cv2.imshow('filtered image', filtered_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

 #       hog_images.append(hog_image)


""" 
#clf = svm.SVC()
hog_features = np.array(hog_features)
y_train_labels =  np.array(y_train_labels).reshape(len(y_train_labels),1)
#y_train_labels=np.array(y_train_labels)
data_frame = np.hstack((hog_features,y_train_labels))
np.random.shuffle(data_frame)

percentage = 75
partition = int(len(hog_features)*percentage/100)

x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()
"""
# clf.fit(x_train,y_train)
"""
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
"""

#y_pred = clf.predict(x_test)

pca = PCA(n_components=60)
principalComponents = pca.fit_transform(filtered_features)

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    principalComponents, y_train_labels, test_size=0.15, random_state=0)

# Feature Scaling
sc = StandardScaler()
"""
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
"""
pc = sc.fit_transform(principalComponents)


# Fitting SVM to the Training set
classifier = SVC(kernel='poly', random_state=0, coef0=0.5)
#from sklearn.model_selection import ShuffleSplit
# n_samples=principalComponents.shape[0]
#cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
scores = cross_val_score(classifier, pc, y_train_labels, cv=10)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


classifier.fit(x_train, y_train)


"""
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
from sklearn.model_selection import cross_val_score
scores=cross_val_score(classifier, x_train, y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

"""

# Predicting the Test set results
y_pred = classifier.predict(x_test)

print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
#print(classification_report(y_test, y_pred))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

"""
correct=0
incorrect=0
for i in range (32):
    for j in range (32):
        if i==j:
            correct=correct+cm[i][j]
        else:
            incorrect=incorrect+cm[i][j]
            
accuracy=correct/(correct+incorrect)*100
print (accuracy)
"""
"""
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""
