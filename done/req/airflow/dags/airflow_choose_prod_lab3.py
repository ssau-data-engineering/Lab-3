from datetime import datetime, timedelta
from airflow.sensors.time_delta import TimeDeltaSensorAsync
from airflow.decorators import task, dag


@dag(
    dag_id='choose_prod_model',
    start_date=datetime(2021, 1, 1),
    schedule_interval=None,
    catchup=False,
    doc_md=__doc__,
)
def choose_prod_model():
    wait_interval = TimeDeltaSensorAsync(
        task_id="wait_interval", 
        delta=timedelta(seconds=10),
    )
    
    @task
    def check_mlflow_models():
        from mlflow import MlflowClient

        client = MlflowClient()
        best_f1_w = -1.0
        best_model = None

        for rm in client.search_registered_models():
            latest_model_version = rm.latest_versions[-1]
            exp_info = client.get_run(latest_model_version.run_id)

            if exp_info.data.metrics['f1_w'] > best_f1_w:
                best_f1_w = exp_info.data.metrics['f1_w']
                best_model = latest_model_version

        if best_model is not None:
            client.transition_model_version_stage(
                name=best_model.name, version=best_model.version, stage="Production"
            )
        else:
            print('Models not found')
    
    wait_interval >> check_mlflow_models()

choose_prod_model_pipeline = choose_prod_model()
