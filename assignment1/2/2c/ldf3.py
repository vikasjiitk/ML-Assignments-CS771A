import glob
from nltk.corpus import stopwords
from textblob import TextBlob
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

stop = stopwords.words('english')
k_fold = 10
folders = [i for i in range(1,k_fold+1)]
scores = []
confusion = np.array([[0, 0], [0, 0]])
for m in range(1,k_fold+1):
    print 'cross validation no: %d'%m
    #TRAINING
    tr_Dict = {
    'label' : [],
    'message' : []
    }
    all_words = []
    no_of_vectors = 0
    for j in range(1,k_fold+1):
        if (m != j):
            tr_source_dir = '../../assgnData/bare/part' + str(j)
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

                filtered_msg = ''
                for word in msg.split():
                    if (word not in stop):
                        filtered_msg = filtered_msg + word + ' '
                msg = filtered_msg

                lemmatized_msg = split_into_lemmas(msg)
                msg = ' '.join(lemmatized_msg)

                tr_Dict['message'].append(msg)

    idfDict = {}
    for i in tr_Dict['message']:
        for word in i.split():
            all_words.append(word)
    # print all_words
    print "Total no. of words: %d"%len(all_words)
    vocabulary = set(all_words)
    # print vocabulary

    Dict = {}
    key = 0
    for i in vocabulary:
        idfDict[key] = 0
        Dict[i] = key
        key += 1
    V = len(vocabulary)
    print "size of vocabulary: %d"%V
    # print no_of_vectors
    X = np.zeros((no_of_vectors,V))
    Y = np.zeros(no_of_vectors)
    Y.reshape(-1,1)
    index = 0
    for i in tr_Dict['label']:
        if i == 'spam':
            Y[index] = 1
        else:
            Y[index] = 0
        index += 1
    index = 0
    # print Y
    # for i in tr_Dict['message']:
    #     for word in i.split():
    #         X[index][Dict[word]] = 1
    #     index += 1
    for i in tr_Dict['message']:
        for word in i.split():
            X[index][Dict[word]] += 1
        maxf = max(X[index])
        for k in range(V):
            if X[index][k] != 0:
                idfDict[k] += 1
                X[index][k] = 0.5 + (0.5/maxf)*X[index][k]
        index += 1
    D = no_of_vectors
    for i in range(V):
        idfDict[i] = np.log(1+D/(1+idfDict[i]))
        for k in range(no_of_vectors):
            X[k][i] = X[k][i]*idfDict[i]

    per = Perceptron()
    per.fit(X,Y)

    # TESTING
    ts_Dict = {
    'label' : [],
    'message' : []
    }
    ts_source_dir = '../../assgnData/bare/part' + str(m)
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

        filtered_msg = ''
        for word in msg.split():
            if (word not in stop):
                filtered_msg = filtered_msg + word + ' '
        msg = filtered_msg

        lemmatized_msg = split_into_lemmas(msg)
        msg = ' '.join(lemmatized_msg)

        ts_Dict['message'].append(msg)

    no_of_test_vectors = len(ts_Dict['label'])
    Xt = np.zeros((no_of_test_vectors,V))
    Yt = np.zeros(no_of_test_vectors)
    Yp = np.zeros(no_of_test_vectors)
    Yt.reshape(-1,1)
    Yp.reshape(-1,1)
    # print Yp
    index = 0
    for i in ts_Dict['label']:
        if i == 'spam':
            Yt[index] = 1
        else:
            Yt[index] = 0
        index += 1
    index = 0

    for i in ts_Dict['message']:
        for word in i.split():
            if word in Dict:
                Xt[index][Dict[word]] += 1
        maxf = max(Xt[index])
        for k in range(V):
            if Xt[index][k] != 0:
                Xt[index][k] = 0.5 + (0.5/maxf)*Xt[index][k]
                Xt[index][k] = Xt[index][k]*idfDict[k]
        index += 1

    for i in range(no_of_test_vectors):
        Yp[i] = per.predict(Xt[i])

    print classification_report(Yt, Yp)
    print "Spam = 1.0 and NonSpam = 0.0"
    confusion += confusion_matrix(Yt, Yp)
    score = f1_score(Yt,Yp, pos_label=1)
    scores.append(score)

# Final Results
print "Final Results: "
print('Total emails classified:', len(tr_Dict['label'])+len(ts_Dict['label']))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)
print '(row=expected, col=predicted)'
