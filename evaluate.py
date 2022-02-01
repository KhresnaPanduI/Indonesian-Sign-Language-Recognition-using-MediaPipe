import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Random forest visualization
df = pd.read_csv('akurasi random forest.csv')

# Change 'rf__n_estimators' to only its value
for i in range(df.shape[0]):
    tree = df['param'][i]
    res = tree.replace("'", "\"")
    res = json.loads(res)
    df.at[i,'param'] = res['rf__n_estimators']

print(df['param'])

sns.set()
ax = sns.lineplot(data=df, x= 'param', y = 'acc')
ax.set(xlabel='Number of Trees', ylabel='Validation Accuracy')
plt.show()
ax.figure.savefig('Random Forest.png')

df_lr = pd.read_csv('akurasi logistic regression.csv')

# Change 'lr__max_iter' to only its value
for i in range(df_lr.shape[0]):
    iter = df_lr['param'][i]
    res = iter.replace("'", "\"")
    res = json.loads(res)
    df_lr.at[i,'param'] = res['lr__max_iter']

#sns.set()
#ax_lr = sns.lineplot(data=df_lr, x= 'param', y = 'acc')
#ax_lr.set(xlabel='Maximum Iterations', ylabel='Validation Accuracy')
#plt.show()
#ax_lr.figure.savefig('Logistic Regression.png')