import datetime
import pendulum

from airflow import DAG
from airflow.decorators import task, dag

import os

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=["data_eng_labs"],
    schedule_interval = '@daily'
)
def a_validate_model_and_move_to_production_mlflow():
    @task.virtualenv(task_id="virtualenv_python", requirements=["mlflow","scikit-learn","boto3"], system_site_packages=False)
    def callable_virtualenv():
        from sklearn.datasets import make_regression
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
        
        import mlflow
        import mlflow.sklearn
        from mlflow import MlflowClient
        from pprint import pprint

        X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        mlflow.set_tracking_uri('http://mlflow_server:5000')
        client = MlflowClient()
        scores = {}
        for rm in client.search_registered_models():
            model = mlflow.sklearn.load_model(model_uri=f"models:/{dict(rm)['name']}/latest")
            scores[dict(rm)['name']] = (mean_squared_error(y_val,model.predict(X_val)), dict(rm)['latest_versions'][0].version)
            client.transition_model_version_stage(
            name=dict(rm)['name'], version=dict(rm)['latest_versions'][0].version, stage="Staging"
            )
        best_model = next(iter(dict(sorted(scores.items(), key=lambda item: item[1]))))
        client.transition_model_version_stage(
            name=best_model, version=scores[best_model][1], stage="Production"
            )
    virtualenv_task = callable_virtualenv()

dag = a_validate_model_and_move_to_production_mlflow()