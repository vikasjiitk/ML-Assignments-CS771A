import glob
from nltk.corpus import stopwords
from textblob import TextBlob
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

stop = stopwords.words('english')

tr_Dict = {
'label' : [],
'message' : []
}
all_words = []
no_of_vectors = 0
tr_source_dir = 'assgnData/bare/part1'
tr_FileList = glob.glob( tr_source_dir + '/*.txt')
for fil in tr_FileList:
    no_of_vectors += 1
    first = len(tr_source_dir)
    if (fil[first+1] == 's'):
        tr_Dict['label'].append('spam')
    else:
        tr_Dict['label'].append('nspam')
    f = open(fil,'r')
    msg = ' '
    for line in f:
        if line != '\n':
            msg = msg + line[:-1] + ' '

# question 1 - part b (removing fluff words) uncomment to run it

    filtered_msg = ''
    for word in msg.split():
        if (word not in stop):
            filtered_msg = filtered_msg + word + ' '
    msg = filtered_msg

# question 1 - part c (lemetizing) uncomment to run it
    lemmatized_msg = split_into_lemmas(msg)
    msg = ' '.join(lemmatized_msg)

    tr_Dict['message'].append(msg)

for i in tr_Dict['message']:
    for word in i.split():
        all_words.append(word)
# print all_words
# print len(all_words)
vocabulary = set(all_words)
# print vocabulary
Dict = {}
key = 0
for i in vocabulary:
    Dict[i] = key
    key += 1
V = len(vocabulary)
# print V
# print no_of_vectors
X = np.zeros((no_of_vectors,V))
Y = np.zeros(no_of_vectors)
index = 0
for i in tr_Dict['label']:
    if i == 'spam':
        Y[index] = 1
    else:
        Y[index] = 0
    index += 1
index = 0
print Y
for i in tr_Dict['message']:
    for word in i.split():
        X[index][Dict[word]] = 1
    index += 1
# print index
# su = 0
# for j in range(no_of_vectors):
#     su = 0
#     for i in X[j]:
#         su +=i
#     print su
clf = LinearDiscriminantAnalysis()
clf.fit(X,Y)

# TESTING
ts_Dict = {
'label' : [],
'message' : []
}
ts_source_dir = 'assgnData/bare/part10'
ts_FileList = glob.glob( ts_source_dir + '/*.txt')
for fil in ts_FileList:
    first = len(ts_source_dir)
    if (fil[first+1] == 's'):
        ts_Dict['label'].append('spam')
    else:
        ts_Dict['label'].append('nspam')
    f = open(fil,'r')
    msg = ' '
    for line in f:
        if line != '\n':
            msg = msg + line[:-1] + ' '
    ts_Dict['message'].append(msg)
no_of_test_vectors = len(ts_Dict['label'])
Xt = np.zeros((no_of_test_vectors,V))
Yt = np.zeros(no_of_test_vectors)
Yp = np.zeros(no_of_test_vectors)
for i in ts_Dict['message']:
    index = 0
    for word in i.split():
        if word in Dict:
            Xt[index][Dict[word]] = 1
    index += 1
for i in range(no_of_test_vectors):
    Yp[i] = clf.predict(Xt[i])
confusion = confusion_matrix(Yt, Yp)
print confusion
