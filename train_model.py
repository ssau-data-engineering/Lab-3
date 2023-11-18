import json
import importlib
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

mlflow.sklearn.autolog()

with open('/opt/airflow/data/lab3/conf.json', 'r') as config_file:
   config_data = json.load(config_file)


x_train = np.asarray(pd.read_csv(f"/opt/airflow/data/lab3/x_train.csv"), dtype=np.float32)
y_train = np.asarray(pd.read_csv(f"/opt/airflow/data/lab3/y_train.csv"), dtype=np.uint8)

x_val = np.asarray(pd.read_csv(f"/opt/airflow/data/lab3/x_val.csv"), dtype=np.float32)
y_val = np.asarray(pd.read_csv(f"/opt/airflow/data/lab3/y_val.csv"), dtype=np.uint8)

tracking_url = 'http://mlflow_server:5000'
mlflow.set_tracking_uri(tracking_url)
# mlflow.set_experiment(experiment_id="0")

for i, config in enumerate(config_data['configs']):
    mlflow.start_run(run_name = config['classificator']) 
    module = importlib.import_module(config['module'])
    classificator = getattr(module, config['classificator'])
    model = classificator(**config['args'])
    model.fit(x_train, y_train)

    prediction = model.predict(x_val)
    signature = infer_signature(x_val, prediction)

    mlflow.log_params(config['args'])
    mlflow.log_metrics({"f1" : f1_score(y_val, prediction, average='weighted')})

    modelInfo = mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path=config['module'], 
        signature=signature, 
        registered_model_name=config['classificator'])

    df = pd.DataFrame({"name":config['classificator'], "uri":modelInfo.model_uri}, index=[i])
    df.to_csv('/opt/airflow/data/lab3/models.csv', mode='a', header=False)
    mlflow.end_run() 