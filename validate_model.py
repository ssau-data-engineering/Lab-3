import os
import operator
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

tracking_url = 'http://mlflow_server:5000'
mlflow.set_tracking_uri(tracking_url)

sk_models = {}
X, y = datasets.load_iris(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=104, test_size=0.8, shuffle=True)

with mlflow.start_run() as run:
    models_file = pd.read_csv("/opt/airflow/data/lab3/models.csv", header=None)
    for model_Info in models_file.iterrows():
        model_uri = model_Info[1][1]
        model_name = model_Info[1][2]
        sk_models[model_uri + " " + model_name] = mlflow.sklearn.load_model(model_uri)

    val_results = {}
    for k, v in sk_models.items():
        prediction = v.predict(X_val)
        val_results[k] = f1_score(y_val, prediction, average="weighted")
    
    best_model = max(val_results.items(), key=operator.itemgetter(1))[0]

    client = MlflowClient()
    result = client.search_model_versions(f"name='{best_model.split(' ')[1]}' and run_id='{best_model.split(' ')[0].split('/')[1]}'")[0]
    client.transition_model_version_stage(name=best_model.split(' ')[1], version=result.version, stage="Production")

    #os.remove("/opt/airflow/data/lab3/models.csv")

