import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.sensors.filesystem import FileSensor

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG(
    dag_id='data-engineering-lab-3-1',
    default_args=default_args,
    description='DAG for data engineering lab 3: training a scikit-learn model',
    schedule_interval=None,
)

track_data = FileSensor(
    task_id='track_data',
    poke_interval=10,
    filepath='/opt/airflow/data/model.json',
    fs_conn_id='filesensor',
    dag=dag,
)

train_model = BashOperator(
    task_id="train_model",
    bash_command="python /opt/airflow/data/train.py",
    dag=dag
)

track_data >> train_model
