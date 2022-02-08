import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from sklearn.metrics import plot_confusion_matrix

# Load test data
X_test = pd.read_csv('X test.csv')
y_test = pd.read_csv('y test.csv')

'''
Random forest evaluation
'''

# load the model from disk.
rf_model = pickle.load(open('RandomForest.pkl', 'rb'))
rf_result = rf_model.score(X_test, y_test)

# Print accuracy and plot confusion matrix
print('Random Forest test accuracy: ', rf_result)

plot_confusion_matrix(rf_model, X_test, y_test)
plt.show()

# Visualization for validation accuracy
df = pd.read_csv('akurasi random forest.csv')

# Change 'rf__n_estimators' to only its value
for i in range(df.shape[0]):
    tree = df['param'][i]
    res = tree.replace("'", "\"")
    res = json.loads(res)
    df.at[i, 'param'] = res['rf__n_estimators']

print(df['param'])

sns.set()
ax = sns.lineplot(data=df, x='param', y='acc')
ax.set(xlabel='Number of Trees', ylabel='Validation Accuracy')
plt.show()
ax.figure.savefig('Random Forest.png')

'''
Logistic Regression Evaluation
'''

# load the model from disk.
lr_model = pickle.load(open('LogisticRegression.pkl.pkl', 'rb'))
lr_result = lr_model.score(X_test, y_test)

# Print accuracy and plot confusion matrix
print('Logistic Regression test accuracy: ', lr_result)

plot_confusion_matrix(lr_model, X_test, y_test)
plt.show()

df_lr = pd.read_csv('akurasi logistic regression.csv')

# Change 'lr__max_iter' to only its value
for i in range(df_lr.shape[0]):
    iter = df_lr['param'][i]
    res = iter.replace("'", "\"")
    res = json.loads(res)
    df_lr.at[i, 'param'] = res['lr__max_iter']

#sns.set()
#ax_lr = sns.lineplot(data=df_lr, x= 'param', y = 'acc')
#ax_lr.set(xlabel='Maximum Iterations', ylabel='Validation Accuracy')
#plt.show()
#ax_lr.figure.savefig('Logistic Regression.png')
