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

tr_images = []
ts_images = []
for i in range(len(tr_im)):
    tr_images.append(np.reshape(tr_im[i],(-1,28)))

for i in range(len(ts_im)):
    ts_images.append(np.reshape(ts_im[i],(-1,28)))

tr_hog = []
for i in range(len(tr_images)):
    tr_hog.append(hog(tr_images[i]))

ts_hog = []
for i in range(len(ts_images)):
    ts_hog.append(hog(ts_images[i]))

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(tr_hog,tr_label)
ts_p = clf.predict(ts_hog)

confusion = np.zeros((10,10))
confusion = confusion_matrix(ts_label, ts_p)
score = f1_score(ts_label, ts_p)

# Final Results
print('Total images labelled:', len(ts_im))
print('Images labelled correctly:', confusion.trace())
print('Score:', score)
print('Confusion matrix:')
print(confusion)
print ('(row=expected, col=predicted)')

# from sklearn.externals.six import StringIO
# with open("mnist.dot", 'w') as f:
#     f = tree.export_graphviz(clf, out_file=f)

# dot -Tpdf mnist.dot -o mnist.pdf
