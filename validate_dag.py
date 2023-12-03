from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator, Mount
from docker.types import DeviceRequest
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import os


default_args = {
    'owner': 'admin',
    'start_date': datetime(2023, 1, 1),
    'retries': 0,
    'dagrun_timeout': timedelta(0, 0, 0, 0, 15, 0, 0)
}

dag = DAG(
    'data_engineering_lab_3_validate_model',
    default_args=default_args,
    description='DAG for data engineering lab 3: validating sklearn classifier',
    schedule_interval='@daily',
)


validate_model = BashOperator(
    task_id="validate_model",
    bash_command="python /opt/airflow/data/lab3/scripts/validate_model.py",
    dag=dag
)

validate_model


if __name__ == "__main__":
    dag.test()