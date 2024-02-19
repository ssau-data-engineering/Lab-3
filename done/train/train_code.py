# Импорт библиотек и модулей
import glob
import json
import os
import numpy as np

import logging
from datetime import datetime, timedelta

from keras.datasets import mnist
from keras.utils import to_categorical

from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier # 1
from sklearn.svm import SVC # 2
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier # 3, 4
from sklearn.tree import DecisionTreeClassifier #5
from sklearn.neighbors import KNeighborsClassifier # 6

#from config_dir import path_to_config, mlflow_server_uri

import mlflow
from mlflow.models import infer_signature

ALL_MODELS = dict(zip(['MLPClassifier', 'SVC', 'HistGradientBoostingClassifier',
                       'RandomForestClassifier', 'DecisionTreeClassifier',
                       'KNeighborsClassifier'],
                      ['MLPClassifier', 'SVC', 'HistGradientBoostingClassifier',
                       'RandomForestClassifier', 'DecisionTreeClassifier',
                       'KNeighborsClassifier'])
                  )


def run_experiment(config, sk_model):
    # Логирование в MLflow 
    logging.getLogger().setLevel(logging.INFO) 

    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    mlflow.set_tracking_uri("http://90.156.229.153:5000") 

    mlflow.set_experiment("MNIST_Experiment")

    with mlflow.start_run() as run:
        # loading the dataset
        (train_X, y_train), (test_X, y_test) = mnist.load_data()
        # Reducing images to 784х1-dimensional and normalizing pixel values ​​to 1
        X_train = train_X.reshape((-1, 28*28)) / 255
        X_test = test_X.reshape((-1, 28*28)) / 255
        # one-hot encoding (we have score [0.95, 0.001, ..., 0.02] get -->[1, 0, ..., 0])
        y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)

    with open(config, 'r') as config_file:
        logging.info(f"Reading data from '{sk_model}'...")
        config_data = json.load(config_file)
        clf_name = config_data['model']

        # logging of the train process
        logging.info(f"Create {config['name']} model")
        model = sk_model.get(clf_name)(**config_data['parameters'])
        model.fit(X_train, np.argmax(y_train, axis=1))

        # Infer the model signature
        logging.info("Performing model prediction...")
        y_pred = model.predict(X_test)
        signature = infer_signature(X_test, y_pred)

        accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
        f1 = f1_score(np.argmax(y_test, axis=1), y_pred, average='macro')

        # metric registration
        logging.info("Log metrics")
        mlflow.log_metric("accuracy:", accuracy)
        mlflow.log_metric("f1-score:", f1)

        # Log the sklearn model and register as version 1
        logging.info("Log model")
        mlflow.sklearn.log_model(
            sk_model=model, # logging the model in the registry
            artifact_path="sklearn-model", 
            signature=signature,
            registered_model_name=config["name"],
        )

def main():
    path_to_configs = glob.glob(os.path.join("/home/masha/Prerequisites/airflow/data/configs", "*.json")) 
    for config in path_to_configs:
        run_experiment(config, ALL_MODELS)

if __name__ == "__main__":
    main()