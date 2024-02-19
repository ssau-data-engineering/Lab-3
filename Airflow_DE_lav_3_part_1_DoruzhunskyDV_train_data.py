from datetime import datetime
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.bash_operator import BashOperator
import os
os.environ["AWS_ACCESS_KEY_ID"] = "minio"   # Данные для сохранения переменны среды
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

default_args = { # аргументы по умолчанию
    'owner': 'airflow', # владелец (можно сортировать в airflow)
    'start_date': datetime(2023, 1, 1), # начало работы
    'retries': 1, # количество повторных попыток, которое должно быть выполнено, прежде чем задача будет провалена
}

dag = DAG(
    'Airflow_DE_lab_3_part_1_train_models', # название DAG
    default_args = default_args, # определение аргументов по умолчанию
    description = 'DAG Training models classificator MLP, KNeighbros, GaussianProcess, RandomForest, AbaBoost', # описание
    schedule_interval = None, # автоматический запуск
)

monitoring = FileSensor( # загрузка файла config.json
    task_id = 'download_json', # название этапа
    poke_interval = 10, #  время, в течение которого задание должно ждать между каждой попыткой
    filepath = '/opt/airflow/data/config.json', # путь по которому происходит поиск файла и название файла
    fs_conn_id = 'connection', # название подключения, необходимо ввести в airflow
    dag = dag, # использование DAG
)

train_data = BashOperator(
    task_id = "train_data_lab_3", # название задания
    bash_command = "python /opt/airflow/data/train_data_lab_3.py", # запуск файла python из данной директории
    dag = dag # использование DAG
)

monitoring >> train_data # последовательность заданий для выполнения, выполняются поочередно