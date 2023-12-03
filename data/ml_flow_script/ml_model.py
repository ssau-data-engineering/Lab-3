import json
import importlib
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import logging

print("hi!!")
# Установка уровня логирования
logging.basicConfig(level=logging.INFO)

# Загрузка конфигурационных данных из JSON файла
with open('/opt/airflow/data/ml_flow_in/config.json', 'r') as config_file:
    config_data = json.load(config_file)

# Загрузка данных и разделение на обучающий и тестовый наборы
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=90, test_size=0.2, shuffle=True)

# Установка URL для отслеживания результатов MLflow
tracking_url = 'http://mlflow_server:5000'
mlflow.set_tracking_uri(tracking_url)

# Установка названия эксперимента
experiment_name = 'experiment_nb'
mlflow.set_experiment(experiment_name)

# Создание списка для данных в DataFrame
df_data = []

# Начало MLflow run
for i, config in enumerate(config_data['configs']):
    # Импорт модуля и создание экземпляра классификатора
    module = importlib.import_module(config['module'])
    classificator = getattr(module, config['classificator'])
    
    # Перебор вариантов параметров
    for param_set in config['param_sets']:
        # Проверка, активен ли уже run
        if mlflow.active_run():
            mlflow.end_run()

        # Создание экземпляра классификатора с текущими параметрами
        model = classificator(**param_set)
        
        # Обучение модели
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        signature = infer_signature(X_test, prediction)
  
        # Логирование параметров и метрик в MLflow
        with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id):
            mlflow.log_params(param_set)
            mlflow.log_metrics({
                "accuracy": accuracy_score(y_test, prediction),
                "f1": f1_score(y_test, prediction, average='weighted'),
                "precision": precision_score(y_test, prediction, average='weighted'),
                "recall": recall_score(y_test, prediction, average='weighted')
            })

            # Логирование модели в MLflow и добавление информации в список
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=config['module'],
                signature=signature,
                registered_model_name=config['classificator']
            )
            df_data.append({"uri": model_info.model_uri, "name": config['classificator']})

            # Логирование в консоль
            logging.info(f"Model {config['classificator']} with parameters {param_set} logged with URI: {model_info.model_uri}")

# Создание DataFrame из списка данных
df = pd.DataFrame(df_data)
print(df)
# Запись DataFrame в CSV файл
df.to_csv('/opt/airflow/data/ml_flow_out/models.csv', mode='w', header=True, index=False)
