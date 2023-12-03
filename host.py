import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__=="__main__":
    mlflow.set_tracking_uri('http://mlflow_server:5000')

    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = []

    with mlflow.start_run() as run:
        models_file = pd.read_csv("/opt/airflow/data/models.csv", header=None)
        for model_Info in models_file.iterrows():
            model_uri = model_Info[1][1]
            model_name = model_Info[1][2]
            models.append(
                {
                    'model_uri': model_uri,
                    'model_name': model_name,
                    'model': mlflow.sklearn.load_model(model_uri)
                }
            )

        results = []
        for el in models:
            res = el['model'].predict(X_test)
            results.append(
                {
                    'model_uri': el['model_uri'],
                    'model_name': el['model_name'],
                    'res': accuracy_score(y_test, res)
                }
            )
        
        best_result = max(results, key=lambda x: x['res'])

        client = MlflowClient()
        result = client.search_model_versions(f'name="{best_result["model_name"]}" and run_id="{best_result["model_uri"]}"')[0]
        client.transition_model_version_stage(name=best_result["model_name"], version=result.version, stage="Production")