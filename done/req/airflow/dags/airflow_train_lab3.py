import os
from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

from docker.types import Mount

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}


dag = DAG(
    'train_nn',
    default_args=default_args,
    description='DAG train NN',
    schedule_interval=None,
)

wait_for_new_file = FileSensor(
    task_id='wait_for_new_train_file',
    poke_interval=10,  # Interval to check for new files (in seconds)
    filepath='/opt/airflow/data/lab3_configs',  # Target folder to monitor
    fs_conn_id='file_train_connection',
    dag=dag,
)

train_nn = DockerOperator(
    task_id='train_nn_on_new_config',
    image='sklearn_image',
    docker_url="tcp://docker:2375", # For Dind usage case
    mount_tmp_dir=False,
    network_mode='host',
#    env_file='/dockerfiles/.env',
    entrypoint='bash',
    command=['-c', "python /data/scripts/train_sklearn.py /data/lab3_configs"],
    mounts=[
        Mount(source='/data', target='/data', type='bind'), 
    ],
    dag=dag,
)


wait_for_new_file >> train_nn