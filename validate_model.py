from functools import partial
import os
import shutil
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import yaml
import logging
import sklearn.metrics as sk_metrics

logging.getLogger().setLevel(logging.INFO)

EXPERIMENT_NAME = "experiment-1"
TARGET_METRIC_NAME = "f1"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio" # "dx2w7NVYkTqM2PogeEWr"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123" # "qlPbFW0NeEwAU7IyFhXut5OgfC7m46UYtVXn9WRc"
mlflow.set_tracking_uri("http://mlflow_server:5000")


logging.info("Preprocess data")
X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_test, X_val, y_test, y_val = train_test_split(
    X, y, test_size=0.5, random_state=42
)

CONFIGS_PATH = "/opt/airflow/data/lab3/configs"
CONFIG_ARCHIVE_PATH = "/opt/airflow/data/lab3/config_archive"

def archive_config(path: str):
    _, config_name = os.path.split(path)
    shutil.move(path, os.path.join(CONFIG_ARCHIVE_PATH, config_name))

def is_equal_experiments(e1, e2):
    if isinstance(e1, dict) and isinstance(e2, dict):
        for k1, v1 in e1.items():
            if k1 in e2 and e2[k1] == str(v1):
                continue
            return False
        return True
    return False


def transition_to_stage(client, run_id, stage):
    model_info = client.search_model_versions(f"run_id='{run_id}'")[0]
    logging.info(f"Moving model {model_info.name} with version {model_info.version} to stage: {stage}")
    client.transition_model_version_stage(name=model_info.name, version=model_info.version, stage=stage)


for config_name in os.listdir(CONFIGS_PATH):
    logging.info(f"Process config {config_name}")
    config_path = os.path.join(CONFIGS_PATH, config_name)
    
    with open(config_path, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)
    
    logging.info("Create metrics")
    metrics = []
    for metric_name, meta in config["metrics"].items():
        if meta is None:
            meta = {}
        metrics.append((metric_name, partial(getattr(sk_metrics, metric_name), **meta)))
    
    logging.info("Retrieve experiment data")
    
    current_experiment= dict(mlflow.get_experiment_by_name(EXPERIMENT_NAME))
    df = mlflow.search_runs([current_experiment["experiment_id"]])
    params = df.filter(like="params")
    params = params.set_index(df["run_id"])
    params.columns = [col.replace("params.", "") for col in params.columns]

    metrics_df = df.filter(like="metrics")
    metrics_df = metrics_df.set_index(df["run_id"])
    metrics_df.columns = [col.replace("metrics.", "") for col in metrics_df.columns]
    
    #logging.info(df.columns)
    #logging.info(metrics_df.columns)
    
    target_metric_val = None
    
    for run_id, exp_params in params.iterrows():
        if is_equal_experiments(config["params"], exp_params.to_dict()):
            logging.info(f"Loading model from mlflow ({run_id=})")
            model_uri = "runs:/{}/model".format(run_id)
            model = mlflow.sklearn.load_model(model_uri)
            
            logging.info("Inference model")
            y_pred = model.predict(X_test) # type: ignore

            with mlflow.start_run(run_id=run_id):
                logging.info("Calcualte metrics")
                metric_vals = [(name, metric(y_test, y_pred)) for name, metric in metrics]
                for metric_name, metric_val in metric_vals:
                    if metric_name.replace("_score", "") == TARGET_METRIC_NAME:
                        target_metric_val = metric_val
                    mlflow.log_metric(metric_name.replace("_score", "") + "_test", metric_val) # type: ignore
            
            # If there'is no tested models
            if len(metrics_df.filter(like="_test")) == 0:
                logging.info("There're no tested models")
                logging.info("Move this model to stage: production")
                
                client = mlflow.client.MlflowClient(mlflow.get_tracking_uri())
                transition_to_stage(client, run_id, "Production")
                
            else:
                logging.info("There're tested models")
                metrics_df = metrics_df.sort_values(by=f"{TARGET_METRIC_NAME}_test", ascending=True)
                best_yet_val = metrics_df[f"{TARGET_METRIC_NAME}_test"].iloc[0]
                if not pd.isna(best_yet_val) and best_yet_val <= target_metric_val:
                    logging.info("Move this model to stage: production")
                    client = mlflow.client.MlflowClient(mlflow.get_tracking_uri())
                
                    transition_to_stage(client, run_id, "Production")

                else:
                    logging.info("This model worse than existing")

            break