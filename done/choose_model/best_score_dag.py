import os
import datetime
import pendulum
import numpy as np
#from pprint import pprint

from airflow.decorators import task, dag
from airflow.sensors.filesystem import FileSensor
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn
from mlflow.client import MlflowClient

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

default_args = {
    "depends_on_past": True,  
}

@dag(
    'airflow_lab3_choose_model',
    schedule=None,  
    default_args=default_args,
    description='DAG: choose the best sklearn classifier',
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
)

def model_selection():

    @task
    def analyze_model_results():
        client = MlflowClient()
        best_model = None
        best_accuracy = -1
        best_f1_score = -1
        for rm in client.search_registered_models():
            model_uri = f"models:/{dict(rm)['name']}/latest"
            runs = client.search_runs(
                experiment_ids=[dict(rm)['experiment_id']],
                filter_string="",
                run_view_type=2,  # ACTIVE_ONLY
                max_results=1,
            )
            if len(runs) > 0:
                metrics = client.get_run(runs[0].info.run_id).data.metrics
                accuracy = metrics.get("accuracy", 0)
                f1 = metrics.get("f1-score", 0)
                if accuracy > best_accuracy:
                    best_model = model_uri
                    best_accuracy = accuracy
                    best_f1_score = f1
        return best_model, best_accuracy, best_f1_score

    @task
    def register_best_model(best_model_info):
        best_model_uri, best_accuracy, best_f1_score = best_model_info
        mlflow.set_tracking_uri("http://90.156.229.153:5000")
        mlflow.set_experiment("MNIST_Experiment")
        with mlflow.start_run() as run:
            mlflow.log_metric("accuracy", best_accuracy)
            mlflow.log_metric("f1-score", best_f1_score)
            mlflow.register_model(
                model_uri=best_model_uri,
                name="best_model",
                await_registration_for=0
            )

    model_results_task = analyze_model_results()
    register_model_task = register_best_model(model_results_task)

    model_results_task >> register_model_task

selection_dag = model_selection()