from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator, Mount
from docker.types import DeviceRequest
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import os

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"



default_args = {
    'owner': 'admin',
    'start_date': datetime(2023, 1, 1),
    'retries': 0,
    'dagrun_timeout': timedelta(0, 0, 0, 0, 15, 0, 0)
}

dag = DAG(
    'data_engineering_lab_3_train_model',
    default_args=default_args,
    description='DAG for data engineering lab 3: training sklearn classifier',
    schedule_interval=None,
)


wait_for_new_configs = FileSensor(
    task_id='wait_for_new_configs',
    poke_interval=5,  # Interval to check for new files (in seconds)
    filepath='/opt/airflow/data/lab3/configs',  # Target folder to monitor
    fs_conn_id='default_conn_id',
    dag=dag,
)

train_model = BashOperator(
    task_id="train_model",
    bash_command="python /opt/airflow/data/lab3/scripts/train_model.py",
    dag=dag
)

wait_for_new_configs >> train_model

if __name__ == "__main__":
    dag.test()