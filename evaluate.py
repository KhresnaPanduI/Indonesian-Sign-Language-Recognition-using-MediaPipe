from sklearn.metrics import ConfusionMatrixDisplay
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

#fig, ax = plt.subplots(figsize=(8,8))
#cmp = ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Blues', ax=ax)
#plt.savefig('Random Forest Confusion Matrix.png', bbox_inches='tight', dpi=300)
#plt.show()

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

# Change 'lr__max_iter' to only its value
for i in range(df_lr.shape[0]):
    iter = df_lr['param'][i]
    res = iter.replace("'", "\"")
    res = json.loads(res)
    df_lr.at[i, 'param'] = res['lr__max_iter']

sns.set()
ax_lr = sns.lineplot(data=df_lr, x='param', y='acc')
ax_lr.set(xlabel='Maximum Iterations', ylabel='Validation Accuracy')
plt.show()
ax_lr.figure.savefig('Logistic Regression.png')
