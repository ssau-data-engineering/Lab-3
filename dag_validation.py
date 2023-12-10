from datetime import datetime
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
import os

os.environ["AWS_ACCESS_KEY_ID"] = "minio" # переменные среды для сохранения модели
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

default_args = { # аргументы по умолчанию
    'owner': 'airflow', # владелец (можно сортировать в airflow)
    'start_date': datetime(2023, 12, 10), # начало работы
    'retries': 1, # количество повторных попыток, которое должно быть выполнено, прежде чем задача будет провалена
}

dag = DAG(
    'validation', # название DAG
    default_args=default_args, # определение аргументов по умолчанию
    description='DAG for validate', # описание DAG
    schedule_interval='@daily', # запуск один раз в день
)

validation = BashOperator(
    task_id="validation", # название задания
    bash_command="python /opt/airflow/data/validation.py", # запуск файла python из данной директории
    dag=dag # использование DAG
)

validation # последовательность заданий для выполнения, выполняются поочередно