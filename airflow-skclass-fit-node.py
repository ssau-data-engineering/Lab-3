from datetime import datetime
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.bash_operator import BashOperator
import os

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG(
    'sklearn_classificator_fit',
    default_args=default_args,
    description='DAG for training sk-learn classificator according to configuration files',
    schedule_interval=None,
)

wait_for_conf_file = FileSensor(
    task_id='wait_for_conf_file',
    poke_interval=10,
    filepath='/opt/airflow/data/lab3/conf.json',
    fs_conn_id='fs_default',
    dag=dag,
)

train_model = BashOperator(
    task_id="train_model",
    bash_command="python /opt/airflow/data/lab3/train_model.py",
    dag=dag
)

wait_for_conf_file >> train_model