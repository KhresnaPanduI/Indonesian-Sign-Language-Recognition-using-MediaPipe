import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('coords.csv')
X = df.drop('class', axis=1)  # features
y = df['class']  # label/target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Create model with sklearn pipeline
rf_pipe = Pipeline([
    ('ss', StandardScaler()),
    ('rf', RandomForestClassifier())
])

lr_pipe = Pipeline([
    ('ss', StandardScaler()),
    ('lr', LogisticRegression())
])

# Initiaze the hyperparameters for each dictionary
param_RF = {}
param_RF['rf__n_estimators'] = [10, 25, 50, 100, 250, 500, 750, 1000]

param_LR = [
    {'lr__penalty': ['l1'],
     'lr__solver': ['liblinear'],
     'lr__max_iter': [50, 100, 250, 500, 750]},
    {'lr__solver': ['lbfgs'],
     'lr__max_iter': [50, 100, 250, 500, 750]}
]

# Creating Gridsearch for each model
gs_lr = GridSearchCV(lr_pipe,
                     param_LR,
                     cv=5,
                     verbose=2,
                     scoring='accuracy')
gs_lr = gs_lr.fit(X_train, y_train)

gs_rf = GridSearchCV(rf_pipe,
                     param_RF,
                     cv=5,
                     verbose=2,
                     scoring='accuracy')
gs_rf = gs_rf.fit(X_train, y_train)

# evaluate model
rf_acc = pd.DataFrame({'param': gs_rf.cv_results_["params"], 'acc': gs_rf.cv_results_["mean_test_score"]})
rf_acc.to_csv('akurasi random forest.csv')
rf_best_parameters = gs_rf.best_params_
rf_best_accuracy = gs_rf.best_score_

lr_acc = pd.DataFrame({'param': gs_lr.cv_results_["params"], 'acc': gs_lr.cv_results_["mean_test_score"]})
lr_acc.to_csv('akurasi logistic regression.csv')
lr_best_parameters = gs_lr.best_params_
lr_best_accuracy = gs_rf.best_score_

print('rf all models accuracy: \n', rf_acc)
print()
print('rf best parameters: ', rf_best_parameters)
print('rf best accuracy: ', rf_best_accuracy)
print('lr all models accuracy: \n', lr_acc)
print()
print('lr best paramaters: ', lr_best_parameters)
print('lr best accuracy: ', lr_best_accuracy)

# save best model
import joblib
joblib.dump(gs_rf.best_estimator_, 'randomForest.pkl')
joblib.dump(gs_lr.best_estimator_, 'logisticRegression.pkl')

# export X_test, y_test as csv to be evaluate later on in evaluate.py
X_test.to_csv('X test.csv')
y_test.to_csv('y test.csv')

