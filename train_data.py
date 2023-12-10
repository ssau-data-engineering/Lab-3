import json
import importlib
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

with open('/opt/airflow/data/config.json', 'r') as config_file: # открытие файла с конфигурацией классификаторов
   config_data = json.load(config_file) # загрузка данных о классификаторах в переменную

breast_cancer = load_breast_cancer() # набор данных о раке молочной железы в штате Висконсин (классификация)

X_train, X_test, y_train, y_test = train_test_split( # разделение данных на тестовую и тренировочную выборку
    breast_cancer.data, breast_cancer.target, test_size = 0.3, random_state = 33 # 70 к 30
)
X_validation, X_test, y_validation, y_test = train_test_split( # разделение данных на тестовую и валидационную выборку
    X_test, y_test, test_size = 0.5, random_state = 33 # 50 к 50
)

pd.DataFrame(X_validation).to_csv('/opt/airflow/data/X_validation.csv', index=False) # запись в файл для второго задания
pd.DataFrame(y_validation).to_csv('/opt/airflow/data/y_validation.csv', index=False) # валидационной выборки

tracking_url = 'http://mlflow_server:5000' # задание ссылки для работы с MLflow
mlflow.set_tracking_uri(tracking_url) # подключение к серверу для работы с MLflow

experiment = mlflow.create_experiment("experiment02") # создание эксперимента
mlflow.set_experiment(experiment_id = experiment) # использование созданного эксперимента

for i, config in enumerate(config_data['configs']): # для каждого классификатора из конфигурационного файла
    mlflow.start_run(run_name = config['classificator'], experiment_id = experiment) # запуск в экперименте
    module = importlib.import_module(config['module']) # импортирование для обработки данных в конфигурационном файле
    classificator = getattr(module, config['classificator']) # название классификатора из данных
    model = classificator(**config['args']) # задание модели обучения через аргументы классификатора конфигурационного файла
    model.fit(X_train, y_train) # обучение модели
    prediction = model.predict(X_test) # предсказание модели
    signature = infer_signature(X_test, prediction) # сигнатура

    mlflow.log_params(config['args']) # логгирование параметров модели
    mlflow.log_metrics({"accuracy": accuracy_score(y_test, prediction), # логгирование метрик
                        "precision": precision_score(y_test, prediction, average = 'weighted'),
                        "recall": recall_score(y_test, prediction, average = 'weighted'),
                        "f1": f1_score(y_test, prediction, average = 'weighted')
                        })

    modelInfo = mlflow.sklearn.log_model( # логгирование модели
        sk_model = model, # использованная модель для логгирования
        artifact_path = config['module'], # модуль с помощью которого распознается модель
        signature = signature, # сигнатура
        registered_model_name = config['classificator']) # название модели

    dataFrame = pd.DataFrame({ 
        "name":config['classificator'], # запись назнания модели
        "uri":modelInfo.model_uri # запись информации о модели
        },
        index=[i])
    dataFrame.to_csv('/opt/airflow/data/models.csv', mode = 'a', header = False) # запись моделей в файл
    mlflow.end_run() # завершение работы