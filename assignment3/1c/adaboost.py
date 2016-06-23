from mnist import MNIST
import sys
import numpy as np
from skimage.feature import hog
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

mnist_dir = '../asgnData/'
mndata = MNIST(mnist_dir)
tr_data = mndata.load_training()
ts_data = mndata.load_testing()
tr_im = tr_data[0][0:10000]
tr_im = np.asarray(tr_im)
tr_label = tr_data[1][0:10000]
ts_im = ts_data[0][0:1000]
ts_im = np.asarray(ts_im)
ts_label = ts_data[1][0:1000]
ts_label = np.asarray(ts_label)
val_im = ts_data[0][1000:2000]
val_im = np.asarray(val_im)
val_label = ts_data[1][1000:2000]
val_label = np.asarray(val_label)

tr_images = []
ts_images = []
val_images = []
for i in range(len(tr_im)):
    tr_images.append(np.reshape(tr_im[i],(-1,28)))

for i in range(len(ts_im)):
    ts_images.append(np.reshape(ts_im[i],(-1,28)))

for i in range(len(val_im)):
    val_images.append(np.reshape(val_im[i],(-1,28)))

tr_hog = []
for i in range(len(tr_images)):
    tr_hog.append(hog(tr_images[i]))

ts_hog = []
for i in range(len(ts_images)):
    ts_hog.append(hog(ts_images[i]))

val_hog = []
for i in range(len(val_images)):
    val_hog.append(hog(val_images[i]))

from sklearn.tree import DecisionTreeClassifier

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
print ('RandomForestClassifier')

# K = 200
# acc_RF = 0
# confusion = np.zeros((10,10))
# for i in range(1,K):
#     clf = RandomForestClassifier(max_depth=6, n_estimators=i)
#     clf.fit(tr_hog,tr_label)
#     val_p = clf.predict(val_hog)
#     confusion = confusion_matrix(val_label, val_p)
#     print (i,confusion.trace())
#     if(confusion.trace() > acc_RF):
#         n_RF = i
#         acc_RF = confusion.trace()
# print (n_RF, acc_RF)

#ADABOOST
from sklearn.ensemble import AdaBoostClassifier
print ('AdaboostClassifier')

K = 200
acc_Ada = 0
confusion = np.zeros((10,10))
for i in range(1,K):
    clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=6), n_estimators=i)
    clf.fit(tr_hog,tr_label)
    val_p = clf.predict(val_hog)
    confusion = confusion_matrix(val_label, val_p)
    print (i,confusion.trace())
    if(confusion.trace() > acc_Ada):
        n_Ada = i
        acc_Ada = confusion.trace()
print (n_Ada, acc_Ada)

# clf_rf = RandomForestClassifier(base_estimator = DecisioTreeClassifier(max_depth=4), n_estimators=n_RF)
# clf_ada = AdaBoostClassifier(base_estimator = DecisioTreeClassifier(max_depth=4), n_estimators=n_ada)
