import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle

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
param_RF['rf__n_estimators'] = [25, 50, 100, 150, 200]

param_LR = {}
param_LR['lr__max_iter'] = [100, 250, 500, 1000]
#param_LR['lr__penalty'] = ['l2', 'none']

# Creating Gridsearch for each model
gs_rf = GridSearchCV(rf_pipe,
                     param_RF,
                     cv=5,
                     verbose=2,
                     scoring='accuracy')
gs_rf = gs_rf.fit(X_train, y_train)

gs_lr = GridSearchCV(lr_pipe,
                     param_LR,
                     cv=5,
                     verbose=2,
                     scoring='accuracy')
gs_lr = gs_lr.fit(X_train, y_train)

# evaluate model

rf_best_parameters = gs_rf.best_params_
rf_best_accuracy = gs_rf.best_score_

lr_best_parameters = gs_lr.best_params_
lr_best_accuracy = gs_rf.best_score_

print('rf best parameters: ', rf_best_parameters)
print('rf best accuracy: ', rf_best_accuracy)
print('lr best paramaters: ', lr_best_parameters)
print('lr best accuracy: ', lr_best_accuracy)

# save best model
import joblib
joblib.dump(gs_rf.best_estimator_, 'randomForest.pkl')
joblib.dump(gs_lr.best_estimator_, 'logisticRegression.pkl')
'''
fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

# Evaluate models
for algo, model in fit_models.items():
    y_pred = model.predict(X_test)
    print(algo, accuracy_score(y_test, y_pred))

# Save models in pickle format
def save_model(name, model):
    with open(name, 'wb') as f:
        pickle.dump(model, f)


save_model('RandomForest.pkl', fit_models['rf'])
save_model('LogisticRegression.pkl', fit_models['lr'])
'''
