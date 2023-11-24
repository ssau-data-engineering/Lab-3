from datetime import datetime
import pendulum

from airflow import DAG
from airflow.operators.python_operator import PythonVirtualenvOperator
from airflow.sensors.filesystem import FileSensor

import os

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

def callable_virtualenv():
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    import yaml
    from importlib import import_module

    config = yaml.load(open('/opt/airflow/data/lab3/config.yml','r'),yaml.Loader)

    mlflow.set_tracking_uri('http://mlflow_server:5000')
    mlflow.set_experiment(experiment_id="1")
    mlflow.autolog()

    mlflow.start_run()
    X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    for model_id in config.keys():
        clf = getattr(import_module(config[model_id]['module']), config[model_id]['classificator'])
        model = clf(**config[model_id]['kwargs'])
        model.fit(X_train, y_train)

        # Infer the model signature
        y_pred = model.predict(X_test)
        signature = infer_signature(X_test, y_pred)
        # mlflow.end_run()
        # run = mlflow.last_active_run()
        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name=config[model_id]['classificator'],
        )

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG(
    'a_fit_model_and_export_to_mlflow',
    default_args=default_args,
    description='DAG for extracting audio, transforming to text, summarizing, and saving as PDF',
    schedule_interval=None,
    tags = ["data_eng_labs"],
)

wait_for_new_config_file = FileSensor( 
    task_id='wait_for_new_config_file',
    poke_interval=10,  # Interval to check for new files (in seconds)
    filepath='/opt/airflow/data/lab3',  # Target folder to monitor
    fs_conn_id='file_connection',
    dag=dag,
)

train_model_export_to_mlflow = PythonVirtualenvOperator(
    task_id='train_model_export_to_mlflow',
    python_callable = callable_virtualenv,
    requirements=["mlflow","scikit-learn","pyyaml","boto3"],
    system_site_packages=False,
    dag=dag,
)

wait_for_new_config_file >> train_model_export_to_mlflow