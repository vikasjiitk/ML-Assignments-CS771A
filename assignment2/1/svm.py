import glob
import pandas
import numpy
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import time

def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

stop = stopwords.words('english')

k_fold = 5
folders = [i for i in range(1,k_fold+1)]
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
for i in range(1,k_fold+1):
    print ('cross validation no: ', i)
    #print i
    #TRAINING
    tr_Dict = {
    'label' : [],
    'message' : []
    }
    for j in range(1,k_fold+1):
        if (i != j):
            tr_source_dir = '../assgnData/bare/part'+str(j)
            tr_FileList = glob.glob( tr_source_dir + '/*.txt')
            for fil in tr_FileList:
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
                # print(msg)
                tr_Dict['message'].append(msg)
    tr_messages = pandas.DataFrame(tr_Dict)

    #Converting to numeric form
    count_vectorizer = CountVectorizer()
    tr_counts = count_vectorizer.fit_transform(tr_messages['message'].values)

    clf = svm.SVC(kernel='linear')           ### For Standard Formulation
    # clf = svm.LinearSVC(loss='hinge')        ### For Hinge-loss Formulation

    tr_targets = tr_messages['label'].values

    start = time.time()
    #training
    clf.fit(tr_counts, tr_targets)
    end = time.time()
    # training time
    print ('time taken for training in seconds: ',(end - start))

    # TESTING
    ts_Dict = {
    'label' : [],
    'message' : []
    }
    ts_source_dir = '../assgnData/bare/part'+str(i)
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

    ts_messages = pandas.DataFrame(ts_Dict)

    #converting to numeric form
    ts_counts = count_vectorizer.transform(ts_messages['message'].values)
    all_predictions = clf.predict(ts_counts)

    # result of each fold of training
    print (classification_report(ts_messages['label'], all_predictions))

    # calculating cumilative results
    confusion += confusion_matrix(ts_messages['label'], all_predictions)
    score = f1_score(ts_messages['label'],all_predictions, pos_label='spam')
    scores.append(score)

# Final Results
print('Total emails classified:', len(tr_Dict['label'])+len(ts_Dict['label']))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)
print ('row=expected, col=predicted')
