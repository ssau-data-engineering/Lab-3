import importlib
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__=="__main__":

    with open('/opt/airflow/data/model.json') as f:
        model_params = json.load(f)

    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mlflow.set_tracking_uri('http://mlflow_server:5000')

    with mlflow.start_run() as run:
        for i, data in enumerate(model_params):
            lib = importlib.import_module(data['source'])
            classifier = getattr(lib, data['classifier'])
            model = classifier(**data['kwargs'])
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            signature = infer_signature(X_test, prediction)

            mlflow.log_params(data['kwargs'])
            mlflow.log_metrics({"accuracy": accuracy_score(y_test, prediction)})

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=data['source'],
                signature=signature,
                registered_model_name=data['classifier']
            )

            df = pd.DataFrame({
                "uri": model_info.model_uri,
                "name": data['classifier']
            }, index=[i])
            df.to_csv('/opt/airflow/data/models.csv', mode='a', header=False)
