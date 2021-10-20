import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('coords.csv')
X = df.drop('class', axis=1)  # features
y = df['class']  # label/target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model with sklearn pipeline
pipelines = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression()),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier())
}

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
save_model('RidgeClassifier.pkl', fit_models['rc'])
save_model('GradientBoosting.pkl', fit_models['gb'])
