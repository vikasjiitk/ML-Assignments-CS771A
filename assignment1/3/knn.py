from mnist import MNIST
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

mnist_dir = '../assgnData/'
mndata = MNIST(mnist_dir)
tr_data = mndata.load_training()
ts_data = mndata.load_testing()

confusion = np.zeros((10,10))

k = int(sys.argv[1])
d_metric = sys.argv[2]

print k
print d_metric

def maxlabel(knn):
    label_counter = {}
    for i in knn:
        if i in label_counter:
            label_counter[i] += 1
        else:
            label_counter[i] = 1
    popular_label = sorted(label_counter, key = label_counter.get, reverse = True)
    top_label = popular_label[:1]
    return top_label

neigh = NearestNeighbors(n_neighbors = k, metric = d_metric)
neigh.fit(tr_data[0])
ts_size = len(ts_data[0])
# ts_size = 100
results = neigh.kneighbors(ts_data[0][0:ts_size], k, return_distance = False)
results_klabels = np.zeros((ts_size,k))

for i in range(ts_size):
    for j in range(k):
        results_klabels[i][j] = tr_data[1][results[i][j]]

results_labels = np.zeros((ts_size,1))
for i in range(ts_size):
    results_labels[i] = maxlabel(results_klabels[i])

true_labels = np.zeros((ts_size,1))
for i in range(ts_size):
    true_labels[i][0] = ts_data[1][i]

print classification_report(true_labels, results_labels)
confusion = confusion_matrix(true_labels, results_labels)
score = f1_score(true_labels, results_labels)

# Final Results
print('Total images labelled:', ts_size)
print('Images labelled correctly:', confusion.trace())
print('Score:', score)
print('Confusion matrix:')
print(confusion)
print '(row=expected, col=predicted)'
