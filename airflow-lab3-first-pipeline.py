from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime

from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
import os

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

def branch_func():
    if (os.path.isfile('/opt/airflow/data/x_train.csv') and os.path.isfile('/opt/airflow/data/y_train.csv')
    and os.path.isfile('/opt/airflow/data/x_val.csv') and os.path.isfile('/opt/airflow/data/y_val.csv') 
    and os.path.isfile('/opt/airflow/data/x_test.csv') and os.path.isfile('/opt/airflow/data/y_test.csv')):
        return ignore.task_id
    else:
        return prepare_data.task_id

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

dag = DAG(
    'airflow_lab3_first_pipeline',
    default_args=default_args,
    description='DAG for training sk-learn classificator according to configuration files',
    start_date = datetime(2023, 11, 19),
    schedule_interval=None,
)

branch_op = BranchPythonOperator(
    task_id='branch_task',
    python_callable=branch_func,
    trigger_rule="all_done",
    dag=dag,
)

prepare_data = BashOperator(
    task_id="prepare_data",
    bash_command="python /opt/airflow/data/lab3/prepare_data.py",
    trigger_rule="all_done",
    dag=dag
)

wait_for_conf_file = FileSensor(
    task_id='wait_for_conf_file',
    poke_interval=10,
    filepath='/opt/airflow/data/lab3/conf.json',
    fs_conn_id='FileSensor',
    dag=dag,
)

ignore = EmptyOperator(
    task_id='ignore',
    trigger_rule="all_done"
)

train_model = BashOperator(
    task_id="train_model",
    bash_command="python /opt/airflow/data/lab3/train_model.py",
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag
)

prepare_data >> wait_for_conf_file >> train_model

wait_for_conf_file >> branch_op 

branch_op >> ignore >> train_model
branch_op >> prepare_data >> train_model
