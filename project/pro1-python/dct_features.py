# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:57:08 2019

@author: Narang
"""

""" HOG features """

#folder = "E:/characterdatabase_munish_mru_extended_final"

# from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from docx import Document
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import numpy as np
import cv2
from skimage import data, exposure
from skimage.feature import hog
import imutils
import matplotlib.pyplot as plt
import os, sys
import joblib
from sklearn.metrics import classification_report, accuracy_score
folder = "C:/users/aknar/documents/image procesing/project/dataset/try1"
onlyfolders = [f for f in os.listdir(folder)]
print("Working with {0} folders".format(len(onlyfolders)))
folder_sequence = []
train_files = []
y_train_labels = []
complete_file_name = []
hog_features = []
hog_images = []
for j in range(0, len(onlyfolders)):
    folder_sequence.append(onlyfolders[j])
    subfolder = folder+"/" + onlyfolders[j]
    print(onlyfolders[j])
    onlyfiles = [f for f in os.listdir(subfolder)]
    # print("Working with {0} images".format(len(onlyfiles)))
    # print("Image examples: ")

    # print(onlyfiles[0])
    for _file in onlyfiles:
        train_files.append(_file)
        complete_file_name.append(subfolder+"/"+_file)
        label_in_file = onlyfolders[j]
        y_train_labels.append(label_in_file)

        #image = data.astronaut()
        filename = subfolder+"/"+_file

        im = cv2.imread(filename)
        """cv2.imshow('image1', im)
        plt.imshow(im, cmap = 'gray', interpolation = 'bicubic')"""
        im = cv2.resize(im, (100, 100))
        gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        image = np.float32(gr)/255.0
        fd = cv2.dct(image)
        a = []
        for i in fd:
            a.extend(i)
        # print(fd[0])
        hog_features.append(a)
        # break
    # break

#clf = svm.SVC()
"""
hog_features = np.array(hog_features)
y_train_labels =  np.array(y_train_labels).reshape(len(y_train_labels),1)
"""
"""
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
principalComponents = pca.fit_transform(hog_features)
"""

"""
#y_train_labels=np.array(y_train_labels)
data_frame = np.hstack((hog_features,y_train_labels))
#data_frame = np.hstack((principalComponents,y_train_labels))
np.random.shuffle(data_frame)

percentage = 70
partition = int(len(hog_features)*percentage/100)
#partition = int(len(principalComponents)*percentage/100)
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

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(hog_features, y_train_labels, test_size = 0.25, random_state = 0)

print(len(hog_features[0]))

pc = hog_features
# Feature Scaling
# sc = StandardScaler()
# pc = sc.fit_transform(pc)
"""
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
"""


#from ensemble_class import EnsembleClassifier


clf1 = SVC(kernel='rbf', random_state=0, coef0=0.5)
clf2 = SVC(kernel='poly', random_state=0, coef0=0.5)
clf3 = SVC(kernel='linear', random_state=0, coef0=0.5)
clf4 = SVC(kernel='sigmoid', random_state=0, coef0=0.5)

clf5 = DecisionTreeClassifier(random_state=0)
clf6 = GaussianNB()
clf7 = BernoulliNB()
# majority voting
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3, clf4, clf5, clf6, clf7])
document = Document()
table = document.add_table(rows=1, cols=9)
i = 0
hdr_cells = table.rows[0].cells
for label in ('train-test split', 'rbf-SVM', 'Poly', 'Linear', 'Sigmoid', 'Decition tree', 'gaussian', 'Bernoulli', 'majority voting'):

    hdr_cells[i].text = label
    i = i+1

row_cells = table.add_row().cells
#print('5-fold cross validation:\n')
row_cells[0].text = '5-fold'
i = 1

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, eclf], ['rbf-SVM', 'Poly', 'Linear', 'Sigmoid', 'Decition tree', 'gaussian', 'Bernoulli', 'majority voting']):

    scores = model_selection.cross_val_score(
        clf, pc, y_train_labels, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" %
          (scores.mean(), scores.std(), label))
    result = str(round(scores.mean(), 2)*100) + \
        "% -- " + str(round(scores.std(), 2))
    row_cells[i].text = result
    i = i+1

document.save('Hog_result_ind.docx')

row_cells = table.add_row().cells
row_cells[0].text = '10-fold'
i = 1

print('10-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, eclf], ['rbf-SVM', 'Poly', 'Linear', 'Sigmoid', 'Decition tree', 'gaussian', 'Bernoulli', 'majority voting']):

    scores = model_selection.cross_val_score(
        clf, pc, y_train_labels, cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" %
          (scores.mean(), scores.std(), label))
    result = str(round(scores.mean(), 2)*100) + \
        "% -- " + str(round(scores.std(), 2))

    row_cells[i].text = result
    i = i+1
document.save('DCT_result_ind.docx')
per = 0.15
x_train, x_test, y_train, y_test = train_test_split(
    pc, y_train_labels, test_size=per, random_state=0)

row_cells = table.add_row().cells
row_cells[0].text = '85:15'
i = 1
print('test:train split:per\n')

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, eclf], ['rbf-SVM', 'Poly', 'Linear', 'Sigmoid', 'Decition tree', 'gaussian', 'Bernoulli', 'majority voting']):

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

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, eclf], ['rbf-SVM', 'Poly', 'Linear', 'Sigmoid', 'Decition tree', 'gaussian', 'Bernoulli', 'majority voting']):

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    result = str(round(accuracy_score(y_test, y_pred), 2)*100)+'%'
    print("Accuracy: "+result + "     " + label)
    print('\n')

    row_cells[i].text = result
    i = i+1
document.save('Hog_result_ind.docx')
per = 0.25
x_train, x_test, y_train, y_test = train_test_split(
    pc, y_train_labels, test_size=per, random_state=0)

row_cells = table.add_row().cells
row_cells[0].text = '75:25'
i = 1
print('test:train split:per\n')

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, eclf], ['rbf-SVM', 'Poly', 'Linear', 'Sigmoid', 'Decition tree', 'gaussian', 'Bernoulli', 'majority voting']):

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

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, eclf], ['rbf-SVM', 'Poly', 'Linear', 'Sigmoid', 'Decition tree', 'gaussian', 'Bernoulli', 'majority voting']):

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    result = str(round(accuracy_score(y_test, y_pred), 2)*100)+'%'
    print("Accuracy: "+result + "     " + label)
    print('\n')

    row_cells[i].text = result
    i = i+1


document.save('hog_result_ind.docx')
row_cells = table.add_row().cells
row_cells[0].text = 'multinomial'
pc = hog_features
sc = MinMaxScaler(feature_range=(0, 1))
pc = sc.fit_transform(pc)
i = 0
clf8 = MultinomialNB()
# 5-fold cross validation
scores = model_selection.cross_val_score(
    clf8, pc, y_train_labels, cv=5, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f) [%s]" %
      (scores.mean(), scores.std(), '5 fold multinomial'))
result = str(round(scores.mean(), 2)*100)+"% -- " + str(round(scores.std(), 2))
# 10-fold cross validation
row_cells[i].text = result
scores = model_selection.cross_val_score(
    clf8, pc, y_train_labels, cv=10, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f) [%s]" %
      (scores.mean(), scores.std(), '10 fold multinomial'))
result = str(round(scores.mean(), 2)*100)+"% -- " + str(round(scores.std(), 2))
i += 1
row_cells[i].text = result
per = 0.15
x_train, x_test, y_train, y_test = train_test_split(
    pc, y_train_labels, test_size=per, random_state=0)


print('test:train split: '+str(per))


clf8.fit(x_train, y_train)
y_pred = clf8.predict(x_test)

result = str(round(accuracy_score(y_test, y_pred), 2)*100)+'%'
print("Accuracy: multinomial 85:15 "+result + "     ")
print('\n')
i += 1
row_cells[i].text = result
per = 0.20
x_train, x_test, y_train, y_test = train_test_split(
    pc, y_train_labels, test_size=per, random_state=0)


print('test:train split: '+str(per))


clf8.fit(x_train, y_train)
y_pred = clf8.predict(x_test)

result = str(round(accuracy_score(y_test, y_pred), 2)*100)+'%'
print("Accuracy: multinomial 80:20 "+result + "     ")
print('\n')
i += 1
row_cells[i].text = result
per = 0.25
x_train, x_test, y_train, y_test = train_test_split(
    pc, y_train_labels, test_size=per, random_state=0)


print('test:train split: '+str(per))


clf8.fit(x_train, y_train)
y_pred = clf8.predict(x_test)

result = str(round(accuracy_score(y_test, y_pred), 2)*100)+'%'
print("Accuracy: multinomial 75:25 "+result + "     ")
print('\n')
i += 1
row_cells[i].text = result
per = 0.30
x_train, x_test, y_train, y_test = train_test_split(
    pc, y_train_labels, test_size=per, random_state=0)


print('test:train split: '+str(per))


clf8.fit(x_train, y_train)
y_pred = clf8.predict(x_test)
print(y_pred)
result = str(round(accuracy_score(y_test, y_pred), 2)*100)+'%'
print("Accuracy: multinomial 70:30 "+result + "     ")
print('\n')
i += 1
row_cells[i].text = result


x_train, x_test, y_train, y_test = train_test_split(
    pc, y_train_labels, test_size=0.1, random_state=0)


print('test:train split: '+str(per))


clf8.fit(x_train, y_train)
joblib.dump(clf8, 'filename.pkl', compress=0, protocol=None, cache_size=None)
print(clf8)
"""
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
"""
"""
from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(random_state=0)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10)
# Fitting SVM to the Training set

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0 , coef0=0.5)
    

from sklearn.model_selection import cross_val_score
scores=cross_val_score(classifier, hog_features, y_train_labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
#print(classification_report(y_test, y_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
"""
