from functools import partial
import importlib
import os
import shutil

import mlflow
import yaml
from mlflow.models import infer_signature
from sklearn import datasets
import sklearn.metrics as sk_metrics
from sklearn.model_selection import train_test_split
import logging

logging.getLogger().setLevel(logging.INFO)

EXPERIMENT_NAME = "experiment-1"
CONFIGS_PATH = "/opt/airflow/data/lab3/configs"
CONFIG_ARCHIVE_PATH = "/opt/airflow/data/lab3/config_archive"

#os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
mlflow.set_tracking_uri("http://mlflow_server:5000")

mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

logging.info("Preparing data")
# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True, as_frame=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_test, X_val, y_test, y_val = train_test_split(
    X, y, test_size=0.5, random_state=42
)

for config_name in os.listdir(CONFIGS_PATH):
    logging.info(f"Start processing config {config_name}")
    config_path = os.path.join(CONFIGS_PATH, config_name)

    logging.info(f"Open config {config_path}")
    with open(config_path, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)

    logging.info(f"Create {config['name']} model")
    
    module = importlib.import_module(config["module"])
    model_cls = getattr(module, config["class"])

    logging.info("Create metrics objects")
    
    metrics = []
    for metric_name, meta in config["metrics"].items():
        if meta is None:
            meta = {}
        metrics.append((metric_name, partial(getattr(sk_metrics, metric_name), **meta)))

    logging.info("Train model")
    # Train the model
    model = model_cls(**config["params"])
    model.fit(X_train, y_train)

    logging.info("Inference model")
    # Predict on the test set
    y_pred = model.predict(X_val)

    logging.info("Calcualte metrics")
    # Calculate metrics
    metric_vals = [(name, metric(y_val, y_pred)) for name, metric in metrics]

    # Start an MLflow run
    logging.info("Start mlflow run")
    with mlflow.start_run():
        # Log the hyperparameters
        logging.info(f"Artifacts URI: {mlflow.get_artifact_uri()}")
        logging.info("Log hyperparameters")
        mlflow.log_params(config["params"])

        # Log the loss metric
        logging.info("Log metrics")
        for metric_name, metric_val in metric_vals:
            mlflow.log_metric(metric_name.replace("_score", "") + "_val", metric_val) # type: ignore

        # Set a tag that we can use to remind ourselves what this run was for
        logging.info("Set tag for the experiment")
        mlflow.set_tag("Training Info", "Basic LR model for iris data")

        # Infer the model signature
        logging.info("Infer signature")
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model
        logging.info("Log model")
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train,
            registered_model_name=config["name"],
        )