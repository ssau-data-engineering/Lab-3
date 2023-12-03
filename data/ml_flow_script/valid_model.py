import mlflow
from mlflow import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score

# Установка URL для отслеживания результатов MLflow
tracking_url = 'http://mlflow_server:5000'
mlflow.set_tracking_uri(tracking_url)

# Загрузка данных iris
iris = load_iris()
X, y = iris.data, iris.target

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Установка Experiment ID
experiment_id = "4"

# Чтение информации о моделях из CSV-файла
models_df = pd.read_csv('/opt/airflow/data/ml_flow_out/models.csv')

# Предположим, что у вас есть столбцы 'uri' и 'name' в CSV-файле
model_uris = models_df['uri']
model_names = models_df['name']

# Загрузка моделей из MLflow и логирование в указанный эксперимент
with mlflow.start_run(experiment_id=experiment_id):
    for model_uri, model_name in zip(model_uris, model_names):
        model = mlflow.sklearn.load_model(model_uri)
        mlflow.sklearn.log_model(sk_model=model, artifact_path=model_name)

# Валидация моделей
test_results = {}
for name in model_names:
    model = mlflow.sklearn.load_model(f"runs:/{mlflow.active_run().info.run_id}/{name}")
    prediction = model.predict(X_test)
    test_results[name] = f1_score(y_test, prediction, average="weighted")

name_best_model = max(test_results, key=test_results.get)

# Ваши дополнительные действия с лучшей моделью
client = MlflowClient()
version = client.search_model_versions(f"name='{name_best_model.split(' ')[0]}' and run_id='{name_best_model.split(' ')[1].split('/')[1]}'")[0].version
client.transition_model_version_stage(name=name_best_model.split(' ')[0], version=version, stage="Production")
mlflow.end_run()
