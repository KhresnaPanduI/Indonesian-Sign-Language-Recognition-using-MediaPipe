from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import joblib

# Load test data
X_test = pd.read_csv('X test.csv')
y_test = pd.read_csv('y test.csv')

# Drop index column
X_test.drop(columns=X_test.columns[0],
            axis=1,
            inplace=True)
y_test.drop(columns=y_test.columns[0],
            axis=1,
            inplace=True)

# Random forest evaluation
# load the model from disk.
rf_model = joblib.load('RandomForest.pkl')
rf_result = rf_model.score(X_test, y_test)

# Print accuracy and plot confusion matrix
print('Random Forest test accuracy: ', rf_result)

fig, ax = plt.subplots(figsize=(8,8))
cmp = ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Blues', ax=ax)
plt.savefig('Random Forest Confusion Matrix.png', bbox_inches='tight', dpi=300)
plt.show()

# Visualization for validation accuracy
df = pd.read_csv('akurasi random forest.csv')

# Change 'rf__n_estimators' to only its value
for i in range(df.shape[0]):
    tree = df['param'][i]
    res = tree.replace("'", "\"")
    res = json.loads(res)
    df.at[i, 'param'] = res['rf__n_estimators']

#sns.set()
#ax = sns.lineplot(data=df, x='param', y='acc')
#ax.set(xlabel='Number of Trees', ylabel='Validation Accuracy')
#plt.show()
#ax.figure.savefig('Random Forest.png')

'''
#Logistic Regression Evaluation
'''

# load the model from disk.
lr_model = joblib.load('LogisticRegression.pkl')
lr_result = lr_model.score(X_test, y_test)

# Print accuracy and plot confusion matrix
print('Logistic Regression test accuracy: ', lr_result)

#fig, ax = plt.subplots(figsize=(8,8))
#cmp = ConfusionMatrixDisplay.from_estimator(lr_model, X_test, y_test, cmap='Blues', ax=ax)
#plt.savefig('Logistic Regression Confusion Matrix.png', bbox_inches='tight', dpi=300)
#plt.show()

df_lr = pd.read_csv('akurasi logistic regression.csv')

# split dataframe one for l1 regularisation another for l2
df_lr1 = df_lr.iloc[:4,:]
df_lr2 = df_lr.iloc[5:,:]

# Change 'lr__max_iter' to only its value for l1
for i in range(df_lr1.shape[0]):
    iter = df_lr1['param'][i]
    res = iter.replace("'", "\"")
    res = json.loads(res)
    df_lr1.at[i, 'param'] = res['lr__max_iter']

# Plot for l1
#sns.set()
#ax_lr = sns.lineplot(data=df_lr1, x='param', y='acc')
#ax_lr.set(xlabel='Maximum Iterations', ylabel='Validation Accuracy')
#plt.show()
#ax_lr.figure.savefig('Logistic Regression l1.png', bbox_inches = 'tight')

# Change 'lr__max_iter' to only its value for l2
for i in range(df_lr2.shape[0]):
    i = i + 5 # because index start from 5
    iter = df_lr2['param'][i]
    res = iter.replace("'", "\"")
    res = json.loads(res)
    df_lr2.at[i, 'param'] = res['lr__max_iter']

# plot for l2
#sns.set()
#ax_lr = sns.lineplot(data=df_lr2, x='param', y='acc')
#ax_lr.set(xlabel='Maximum Iterations', ylabel='Validation Accuracy')
#plt.show()
#ax_lr.figure.savefig('Logistic Regression l2.png', bbox_inches = 'tight')

# function to calculate TP, FP, TN, FN
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return(TP, FP, TN, FN)

# confusion matrix for random forest model
y_pred = rf_model.predict(X_test)
rf_cm = confusion_matrix(y_test, y_pred)
#print(rf_cm)
print('==============================')
print('Akurasi Random Forest')
print(rf_cm.diagonal()/rf_cm.sum(axis=1))

for i in range(0, len(rf_cm[0])):
    print('Akurasi kelas-{} = {}'.format(i+1, rf_cm[i][i]/sum(rf_cm[i])))

import numpy as np
FP = rf_cm.sum(axis=0) - np.diag(rf_cm)
FN = rf_cm.sum(axis=1) - np.diag(rf_cm)
TP = np.diag(rf_cm)
TN = rf_cm.sum() - (FP + FN + TP)

print('rf_cm.sum: ', rf_cm.sum())
print('sum(rf_cm): ', sum(rf_cm))
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

print('True Positive: ', TP)
print('False Positive: ', FP)
print('True Negative: ', TN)
print('False Negative: ', FN)

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, digits=4))

# confusion matrix for logistic regression model

lr_y_pred = lr_model.predict(X_test)
lr_cm = confusion_matrix(y_test, lr_y_pred)
#print(lr_cm)
print('==============================')
print('Akurasi Logistic Regression')
print(lr_cm.diagonal()/lr_cm.sum(axis=1))

for i in range(0, len(lr_cm[0])):
    print('Akurasi kelas-{} = {}'.format(i+1, lr_cm[i][i]/sum(lr_cm[i])))

print('jumlah benar kelas 19', lr_cm[18][18])
print('jumlah total kelas 19', sum(lr_cm[18]))
print('total/jumlah benar: ', lr_cm[18][18]/sum(lr_cm[18]))