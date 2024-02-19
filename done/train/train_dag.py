from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime

default_args = {
    'owner': 'airflow_lab3_task1',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG(
    'train_dag',
    default_args=default_args,
    description='DAG: training sklearn classifier',
    schedule_interval=None,
)

wait_for_new_file = FileSensor(
    task_id='wait_for_new_train_file',
    poke_interval=10,
    filepath='/home/masha/Prerequisites/airflow/data/configs',
    fs_conn_id='file_train_connection',
    dag=dag,
)

result_data_path = "/home/masha/Prerequisites/airflow/data/predict/result__{{ ds }}.json"

train_nn = BashOperator(
    task_id='train_nn_on_new_config',
    bash_command='python /home/masha/Prerequisites/airflow/dags/train_code.py && echo "Results saved to {}"'.format(result_data_path),
    dag=dag,
)

wait_for_new_file >> train_nn

