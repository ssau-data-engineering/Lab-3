import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash_operator import BashOperator

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG(
    dag_id='data-engineering-lab-3-2',
    default_args=default_args,
    description='DAG for data engineering lab 3: hosting the best scikit-learn model'
)

validate_model = BashOperator(
    task_id="validate_model",
    bash_command="python /opt/airflow/data/host.py",
    dag=dag
)

validate_model
