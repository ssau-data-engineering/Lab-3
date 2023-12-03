from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.sensors.filesystem import FileSensor
import os

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

# Определите параметры DAG
default_args = {
    'owner': 'bogdann',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'air_valid_lab3',
    default_args=default_args,
    description='DAG for validate and host sk-learn model',
    schedule_interval=None,
)


validate_model = BashOperator(
    task_id="validate_model",
    bash_command="python /opt/airflow/data/ml_flow_script/valid_model.py",
    dag=dag
)

validate_model


