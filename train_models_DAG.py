from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor

import json
import importlib
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_digits

with DAG(
    'train_models',
    start_date=datetime(2023, 12, 12),
    schedule_interval=None,
    catchup=False
    ) as dag:

    def train_models():
        with open('/opt/airflow/data/config.json', 'r') as config_json:
            config = json.load(config_json)

        digits_df = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(digits_df.data, digits_df.target, test_size = 0.3, random_state = 1)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 1)

        X_val_df = pd.DataFrame(X_val)
        X_val_df.to_csv('/opt/airflow/data/X_val.csv', index=False)
        y_val_df = pd.DataFrame(y_val)
        y_val_df.to_csv('/opt/airflow/data/y_val.csv', index=False)

        mlflow.set_tracking_uri('http://mlflow_server:5000')

        experiment = mlflow.create_experiment("1111111")

        mlflow.set_experiment(experiment_id = experiment)

        for i, config in enumerate(config['configs']):

            cur_classificator = config['classificator']
            cur_module = config['module']
            cur_kwargs = config['kwargs']

            mlflow.start_run(run_name = cur_classificator, experiment_id = experiment)

            module = importlib.import_module(cur_module)
            classificator = getattr(module, cur_classificator)

            model = classificator(**cur_kwargs)
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            signature = infer_signature(X_test, prediction)

            mlflow.log_params(cur_kwargs)
            mlflow.log_metrics({"accuracy": accuracy_score(y_test, prediction),
                                "precision": precision_score(y_test, prediction, average = 'weighted'),
                                "recall": recall_score(y_test, prediction, average = 'weighted'),
                                "f1": f1_score(y_test, prediction, average = 'weighted')
                                })

            model_info = mlflow.sklearn.log_model(
                sk_model = model, 
                artifact_path = cur_module, 
                signature = signature, 
                registered_model_name = cur_classificator)

            dataFrame = pd.DataFrame({
                "name":cur_classificator,
                "uri":model_info.model_uri
                },
                index=[i])
            dataFrame.to_csv('/opt/airflow/data/res_models.csv', mode = 'a', header = False)
            mlflow.end_run()



    wait_config = FileSensor(
        task_id = 'wait_config',
        poke_interval = 20,
        filepath = '/opt/airflow/data',
        fs_conn_id = 'connection_for_lab2',
        dag = dag,
    )

    train_models = PythonOperator(
        task_id = "train_models",
        python_callable=train_models,
        dag=dag
    )

    wait_config >> train_models