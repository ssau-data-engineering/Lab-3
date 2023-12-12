import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from mlflow import MlflowClient
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG(
    'analyze_models',
    start_date=datetime(2023, 12, 12),
    schedule_interval=None,
    catchup=False
    ) as dag:

    def analyze_models():
        mlflow.set_tracking_uri('http://mlflow_server:5000')

        mlflow.set_experiment("1111111")

        X_val = np.asarray(pd.read_csv(f"/opt/airflow/data/X_val.csv"), dtype = np.float32)
        y_val = pd.read_csv(f"/opt/airflow/data/y_val.csv")

        list_models = {}
        mlflow.start_run(run_name = "Best model")
        models = pd.read_csv("/opt/airflow/data/res_models.csv", header = None)

        for model_info in models.iterrows():
            name_model = model_info[1][1]
            uri_model = model_info[1][2]
            list_models[name_model + " " + uri_model] = mlflow.sklearn.load_model(uri_model)

        results = {}
        for i, j in list_models.items():
            prediction = j.predict(X_val)
            results[i] = f1_score(y_val, prediction, average="weighted")

        best_model = max(results, key=results.get)

        version = MlflowClient().search_model_versions(f"name = '{best_model.split(' ')[0]}' and run_id = '{best_model.split(' ')[1].split('/')[1]}'")[0].version
        MlflowClient().transition_model_version_stage(name = best_model.split(' ')[0], version = version, stage = "Production")
        mlflow.end_run()

    analyze_models = PythonOperator(
        task_id = "analyze_models",
        python_callable=analyze_models,
        dag=dag
    )

    analyze_models