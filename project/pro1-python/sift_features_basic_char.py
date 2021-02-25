# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:07:36 2018

@author: Narang
"""

# %matplotlib inline
from sklearn.model_selection import train_test_split
from docx import Document
# from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from skimage import data, exposure
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import MiniBatchKMeans

#from feature_aggregation import BagOfWords, FisherVectors
#from sklearn import BaseEstimator


def gen_rootsift_features(gray_img):
    eps = 1e-7
    # sift = cv2.SIFT_create()
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, descs = sift.detectAndCompute(gray_img, None)
    #descs /= (descs.sum(axis=1, keepdims=True) + eps)
    #descs = np.sqrt(descs)
    return kp, descs


def show_sift_features(gray_img, color_img, kp):
    # return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))


def show_rgb_img(img):
    """Convenience function to display a typical color image"""
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))


def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray


def explain_keypoint(kp):
    print('angle\n', kp.angle)
    print('\nclass_id\n', kp.class_id)
    print('\noctave (image scale where feature is strongest)\n', kp.octave)
    print('\npt (x,y)\n', kp.pt)
    print('\nresponse\n', kp.response)
    print('\nsize\n', kp.size)


def match_sift():
    im1 = cv2.imread("octopus_far_front.jpg")
    gr1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    #image = gr
    kp1, desc1 = gen_rootsift_features(gr1)
    im2 = cv2.imread("octopus_far_offset.jpg")
    gr2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    gr2np = np.array(gr2)
    #image = gr
    kp2, desc2 = gen_rootsift_features(gr2)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda m: m.distance)
    print(len(matches))
    img3 = cv2.drawMatches(gr1, kp1, gr2np, kp2,
                           matches[:18], gr2np.copy(), flags=2)
    plt.imshow(img3), plt.show()


def BOVW(feature_descriptors, n_clusters=150):
    print("Bag of visual words with {} clusters".format(n_clusters))
    # take all features and put it into a giant list
    combined_features = np.vstack(np.array(feature_descriptors))
    # train kmeans on giant list
    print("Starting K-means training")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                             random_state=0).fit(combined_features)
    print("Finished K-means training, moving on to prediction")
    # number of images x number of clusters. initiate matrix of histograms
    bovw_vector = np.zeros([len(feature_descriptors), n_clusters])
    # sift descriptors in each image
    for index, features in enumerate(feature_descriptors):
        try:
            for i in kmeans.predict(features):  # get label for each centroid
                # create individual histogram vector
                bovw_vector[index, i] += 1
        except:
            pass
    return bovw_vector  # this should be our histogram


"""
print 'OpenCV Version (should be 3.1.0, with nonfree packages installed, for this tutorial):'
print(cv2.__version__)
"""
# I cropped out each stereo image into its own file.
# You'll have to download the images to run this for yourself
folder = "C:/Users/aknar/Documents/image procesing/project/dataset/try1"

#folder = "E:/characterdatabese_munish_MRU_originals_copyfrom here to mru_extended/individual characters"
onlyfolders = [f for f in os.listdir(folder)]
print("Working with {0} folders".format(len(onlyfolders)))
folder_sequence = []
train_files = []
y_train_labels = []
complete_file_name = []
sift_features = []
sift_images = []
hist1 = []
h = 0

for j in range(0, len(onlyfolders)):
    folder_sequence.append(onlyfolders[j])
    subfolder = folder+"/" + onlyfolders[j]
    onlyfiles = [f for f in os.listdir(subfolder)]
    print("Working with {0} images".format(len(onlyfiles)))
    print("Image examples: ")

    print(onlyfiles[0])
    img_in_folder = 1
    for _file in onlyfiles:
        train_files.append(_file)
        complete_file_name.append(subfolder+"/"+_file)
        label_in_file = onlyfolders[j][0]
        y_train_labels.append(label_in_file)

        #image = data.astronaut()
        filename = subfolder+"/"+_file
        # filename="octopus_far_front.jpg"

        im = cv2.imread(filename)
        gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #image = gr
        gr_kp, gr_desc = gen_rootsift_features(gr)

        sift_features.append(gr_desc)
        """
        if img_in_folder==1 :
            
            # code to plot character image and its interest points
            
           show_sift_features(gr, im, gr_kp)
           gr_desc[0]
           plt.imshow(gr_desc[0].reshape(16,8), interpolation='none')
           img_in_folder=img_in_folder+1
           
       """
        # match_sift()


# print(complete_file_name[2597])
n_clusters = 300
histogram = BOVW(sift_features, n_clusters)
"""
import matplotlib.pyplot as plt
plt.plot(histogram[0], 'o')
plt.ylabel('frequency');
plt.xlabel('features');
"""

sc = StandardScaler()
"""
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
"""
pc = sc.fit_transform(histogram)
"""
from sklearn.decomposition import PCA
pca = PCA(n_components=60)
principalComponents = pca.fit_transform(histogram)
"""
# X=np.array(sift_features)

# Feature Scaling


"""
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
"""
# Fitting SVM to the Training set

"""
from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', random_state = 0 , coef0=0.5)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=15, metric='minkowski', p=2)

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
classifier = BaggingClassifier(SVC(kernel = 'rbf', random_state = 0 , coef0=0.5),max_samples=0.5, max_features=0.5)

from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(random_state=0)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10)
"""
"""
from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier(n_estimators=100, base_estimator=SVC)
"""
"""
from sklearn.svm import SVC
classifier = SVC(kernel = 'sigmoid', random_state = 0 , coef0=0.5)

from sklearn.model_selection import cross_val_score, cross_val_predict
scores=cross_val_score(classifier, pc, y_train_labels, cv=10  )

print("SVC sigmoid Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
"""

"""
from sklearn.model_selection import cross_val_score, cross_val_predict
scores=cross_val_score(classifier, histogram, y_train_labels, cv=15  )

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
"""
# Fitting SVM to the Training set
# Predicting the Test set results

#from ensemble_class import EnsembleClassifier
#from mlxtend.classifier import EnsembleVoteClassifier


clf3 = SVC(kernel='rbf', random_state=0, coef0=0.5)
clf4 = SVC(kernel='poly', random_state=0, coef0=0.5)
clf5 = SVC(kernel='linear', random_state=0, coef0=0.5)
clf6 = SVC(kernel='sigmoid', random_state=0, coef0=0.5)

clf7 = DecisionTreeClassifier(random_state=0)
clf1 = GaussianNB()
clf2 = BernoulliNB()
# majority voting
#eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3, clf4, clf5, clf6, clf7])
document = Document()
table = document.add_table(rows=1, cols=9)
i = 0
hdr_cells = table.rows[0].cells
for label in ('train-test split', 'gaussian', 'Bernoulli', 'rbf-SVM', 'Poly', 'Linear', 'Sigmoid', 'Decision tree'):

    hdr_cells[i].text = label
    i = i+1

row_cells = table.add_row().cells
#print('5-fold cross validation:\n')
row_cells[0].text = '5-fold'
i = 1
for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7], ['gaussian', 'Bernoulli', 'rbf-SVM', 'Poly', 'Linear', 'Sigmoid', 'Decision tree']):

    scores = cross_validation.cross_val_score(
        clf, pc, y_train_labels, cv=10, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.2f) [%s]" %
          (scores.mean(), scores.std(), label))
    result = str(round(scores.mean(), 2)*100) + \
        "% -- " + str(round(scores.std(), 2))
    row_cells[i].text = result
    i = i+1

row_cells = table.add_row().cells
row_cells[0].text = '10-fold'
i = 1
print('10-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7], ['gaussian', 'Bernoulli', 'rbf-SVM', 'Poly', 'Linear', 'Sigmoid', 'Decision tree']):

    scores = cross_validation.cross_val_score(
        clf, pc, y_train_labels, cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" %
          (scores.mean(), scores.std(), label))
    result = str(round(scores.mean(), 2)*100) + \
        "% -- " + str(round(scores.std(), 2))

    row_cells[i].text = result
    i = i+1

per = 0.15
x_train, x_test, y_train, y_test = train_test_split(
    pc, y_train_labels, test_size=per, random_state=0)

row_cells = table.add_row().cells
row_cells[0].text = '85:15'
i = 1
print('test:train split:per\n')

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7], ['gaussian', 'Bernoulli', 'rbf-SVM', 'Poly', 'Linear', 'Sigmoid', 'Decision tree']):

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    result = str(round(accuracy_score(y_test, y_pred), 2)*100)+'%'
    print("Accuracy: "+result + "     " + label)
    print('\n')

    row_cells[i].text = result
    i = i+1
per = 0.20
x_train, x_test, y_train, y_test = train_test_split(
    pc, y_train_labels, test_size=per, random_state=0)

row_cells = table.add_row().cells
row_cells[0].text = '80:20'
i = 1
print('test:train split:per\n')

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7], ['gaussian', 'Bernoulli', 'rbf-SVM', 'Poly', 'Linear', 'Sigmoid', 'Decision tree']):

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    result = str(round(accuracy_score(y_test, y_pred), 2)*100)+'%'
    print("Accuracy: "+result + "     " + label)
    print('\n')

    row_cells[i].text = result
    i = i+1

per = 0.25
x_train, x_test, y_train, y_test = train_test_split(
    pc, y_train_labels, test_size=per, random_state=0)

row_cells = table.add_row().cells
row_cells[0].text = '75:25'
i = 1
print('test:train split:per\n')

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7], ['gaussian', 'Bernoulli', 'rbf-SVM', 'Poly', 'Linear', 'Sigmoid', 'Decision tree']):

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    result = str(round(accuracy_score(y_test, y_pred), 2)*100)+'%'
    print("Accuracy: "+result + "     " + label)
    print('\n')

    row_cells[i].text = result
    i = i+1
per = 0.30
x_train, x_test, y_train, y_test = train_test_split(
    pc, y_train_labels, test_size=per, random_state=0)

row_cells = table.add_row().cells
row_cells[0].text = '70:30'
i = 1
print('test:train split:per\n')

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7], ['gaussian', 'Bernoulli', 'rbf-SVM', 'Poly', 'Linear', 'Sigmoid', 'Decision tree']):

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    result = str(round(accuracy_score(y_test, y_pred), 2)*100)+'%'
    print("Accuracy: "+result + "     " + label)
    print('\n')

    row_cells[i].text = result
    i = i+1


document.save('Sift_result_all.docx')
"""                      
bow = BagOfWords(100)
for i in range(2):
    for j in range(0, len(features), 10):
        bow.partial_fit(features[j:j+10])
faces_bow = bow.transform(features)
"""
