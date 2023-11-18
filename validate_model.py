import os
import operator
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

tracking_url = 'http://mlflow_server:5000'
mlflow.set_tracking_uri(tracking_url)

x_test = np.asarray(pd.read_csv(f"/opt/airflow/data/lab3/x_test.csv"), dtype=np.float32)
y_test = pd.read_csv(f"/opt/airflow/data/lab3/y_test.csv")

sk_models = {}

mlflow.start_run(run_name = "validate model")
models_file = pd.read_csv("/opt/airflow/data/lab3/models.csv", header=None)
for model_Info in models_file.iterrows():
    name = model_Info[1][1]
    uri = model_Info[1][2]
    sk_models[name + " " + uri] = mlflow.sklearn.load_model(uri)

test_results = {}
for name, model in sk_models.items():
    prediction = model.predict(x_test)
    test_results[name] = f1_score(y_test, prediction, average="weighted")

name_best_model = max(test_results, key=test_results.get)

client = MlflowClient()
version = client.search_model_versions(f"name='{name_best_model.split(' ')[0]}' and run_id='{name_best_model.split(' ')[1].split('/')[1]}'")[0].version
client.transition_model_version_stage(name=name_best_model.split(' ')[0], version=version, stage="Production")
mlflow.end_run()