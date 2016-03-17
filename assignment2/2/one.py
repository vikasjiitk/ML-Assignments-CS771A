import pandas as pd
from sklearn import svm
import numpy
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

def most_common(lst):
    return max(set(lst), key=lst.count)

def most_pred(all_predictions):
    pred = []
    rand = 0
    for i in range(len(all_predictions)):
        l = all_predictions[i]
        if (l[0] != l[1]):
            if(l[1] != l[2]):
                if(l[2] != l[0]):
                    rand += 1
                    pred.append(most_common(all_predictions[i]))
                else:
                    pred.append(most_common(all_predictions[i]))
            else:
                pred.append(most_common(all_predictions[i]))
        else:
            pred.append(most_common(all_predictions[i]))
    print (rand)
    return pred

data = pd.read_csv('../assgnData/connect-4.csv', sep=',',header=None)

#no_vec = len(data)
no_vec = 500

k_fold = 5
confusion = numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

dt = [[0 for j in range(42*3)] for i in range(no_vec)]
dt_label = [0 for i in range(no_vec)]
for i in range(no_vec):
    if(data.loc[i,42]=='win'):
        dt_label[i] = 1
    elif(data.loc[i,42]=='loss'):
        dt_label[i] = -1
    elif(data.loc[i,42]=='draw'):
        dt_label[i] = 0

    for j in range(42):
        if(data.loc[i,j]=='o'):
            dt[i][3*j] = 1
        elif(data.loc[i,j]=='b'):
            dt[i][3*j+1] = 1
        elif(data.loc[i,j]=='x'):
            dt[i][3*j+2] = 1
# print(dt_label)

for i in range(k_fold):
    ti = int(i*no_vec/k_fold)
    tf = int((i+1)*no_vec/k_fold)
    ts_dataw = [] #win
    ts_datal = [] #loss
    ts_datad = [] #draw
    tr_dataw = []
    tr_datal = []
    tr_datad = []
    ts_lw = [] #win
    ts_ll = [] #loss
    ts_ld = [] #draw
    tr_lw = []
    tr_ll = []
    tr_ld = []
    for j in range(no_vec):
        if(j>=ti and j<tf):
            if(dt_label[j]==1):
                ts_dataw.append(dt[j])
                ts_lw.append(1)
            elif(dt_label[j]==-1):
                ts_datal.append(dt[j])
                ts_ll.append(-1)
            elif(dt_label[j]==0):
                ts_datad.append(dt[j])
                ts_ld.append(0)
        else:
            if(dt_label[j]==1):
                tr_dataw.append(dt[j])
                tr_lw.append(1)
            elif(dt_label[j]==-1):
                tr_datal.append(dt[j])
                tr_ll.append(-1)
            elif(dt_label[j]==0):
                tr_datad.append(dt[j])
                tr_ld.append(0)
    # print (tr_lw)
    # print (tr_ll)
    # print (tr_ld)

    clf1 = svm.SVC(kernel='linear') # Win-Loss
    clf1.fit(tr_dataw + tr_datal, tr_lw + tr_ll)

    clf2 = svm.SVC(kernel='linear') # Win-Draw
    clf2.fit(tr_dataw + tr_datad, tr_lw + tr_ld)

    clf3 = svm.SVC(kernel='linear') # Draw-Loss
    clf3.fit(tr_datad + tr_datal, tr_ld + tr_ll)

    true_labels = ts_lw + ts_ll + ts_ld
    all_predictions = [[0,0,0] for j in range(ti,tf)]

    pred1 = clf1.predict(ts_dataw + ts_datal + ts_datad)
    for j in range(0,tf-ti):
        all_predictions[j][0] = pred1[j]

    pred2 = clf2.predict(ts_dataw + ts_datal + ts_datad)
    for j in range(0,tf-ti):
        all_predictions[j][1] = pred2[j]

    pred3 = clf3.predict(ts_dataw + ts_datal + ts_datad)
    for j in range(0,tf-ti):
        all_predictions[j][2] = pred3[j]

    final_pred = most_pred(all_predictions)
    # print (all_predictions)
    print (classification_report(true_labels, final_pred))
    confusion += confusion_matrix(true_labels, final_pred)

print('Confusion matrix:')
print(confusion)
print ('row=expected, col=predicted')
