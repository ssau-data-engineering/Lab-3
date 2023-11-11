import json
import importlib
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

with open('/opt/airflow/data/lab3/conf.json', 'r') as config_file:
   config_data = json.load(config_file)

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104, test_size=0.2, shuffle=True)

tracking_url = 'http://mlflow_server:5000'
mlflow.set_tracking_uri(tracking_url)

with mlflow.start_run() as run:
   for i, config in enumerate(config_data['configs']):
      module = importlib.import_module(config['module'])
      classificator = getattr(module, config['classificator'])
      model = classificator(**config['kwargs'])
      model.fit(X_train, y_train)
      prediction = model.predict(X_test)
      signature = infer_signature(X_test, prediction)
  
      mlflow.log_params(config['kwargs'])
      mlflow.log_metrics({"accuracy": accuracy_score(y_test, prediction),
                        "f1": f1_score(y_test, prediction, average='weighted'),
                        "precision": precision_score(y_test, prediction, average='weighted'),
                        "recal": recall_score(y_test, prediction, average='weighted')})

      modelInfo = mlflow.sklearn.log_model(
         sk_model=model, 
         artifact_path=config['module'], 
         signature=signature, 
         registered_model_name=config['classificator'])
   
      df = pd.DataFrame({"uri":modelInfo.model_uri,"name":config['classificator']}, index=[i])
      df.to_csv('/opt/airflow/data/lab3/models.csv', mode='a', header=False)

