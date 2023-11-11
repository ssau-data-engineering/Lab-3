from datetime import datetime
from airflow import DAG
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
    'sklearn_classificator_host',
    default_args=default_args,
    description='DAG for validate and host sk-learn model',
    schedule_interval='@hourly',
)

validate_model = BashOperator(
    task_id="validate_model",
    bash_command="python /opt/airflow/data/lab3/validate_model.py",
    dag=dag
)

validate_model