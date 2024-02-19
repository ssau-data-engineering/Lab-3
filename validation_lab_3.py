import os
import operator
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.metrics import f1_score

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

tracking_uri = 'http://mlflow_server:5000' # задание ссылки для работы с MLflow
mlflow.set_tracking_uri(tracking_uri) # подключение к серверу для работы с MLflow
mlflow.set_experiment("experiment_6_Doruzhinsky") # использование созданного эксперимента

X_validation = np.asarray(pd.read_csv(f"/opt/airflow/data/X_validation.csv"), dtype = np.float32) # считывание данных из файла валидационной выборки
y_validation = pd.read_csv(f"/opt/airflow/data/y_validation.csv")

list_models = {} # задание пустого объекта(модели)
mlflow.start_run(run_name = "Production model") # запуск в экперименте Production model
models = pd.read_csv("/opt/airflow/data/models.csv", header = None) # считывание моделей
for model_Info in models.iterrows(): # для каждой модели считывается записанное название и информация о модели
    name_model = model_Info[1][1]
    uri_model = model_Info[1][2]
    list_models[name_model + " " + uri_model] = mlflow.sklearn.load_model(uri_model) # запись моделей

results = {} # задание пустого объекта(результат)
for i, j in list_models.items(): # посчет метрики f1 для каждой модели
    pred = j.predict(X_validation)
    results[i] = f1_score(
        y_validation,
        pred,
        average="weighted"
        )

best_model = max(results, key=results.get) # из всех моделей выбиратся та, в которой метрика f1 максимальна
# запись лучшей версии и переопределение stage у лучшей версии с None на Production
version = MlflowClient().search_model_versions(f"name = '{best_model.split(' ')[0]}' and run_id = '{best_model.split(' ')[1].split('/')[1]}'")[0].version
MlflowClient().transition_model_version_stage(name = best_model.split(' ')[0], version = version, stage = "Production")
mlflow.end_run() # завершение работы