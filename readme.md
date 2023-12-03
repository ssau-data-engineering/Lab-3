# Лабораторная работа №3

# Пайплайн обучения модели

В качестве данных был взят датасет iris из библиотеки scikit-learn, в обучении и валидации выборки формируются следующим образом:

    X, y = datasets.load_iris(return_X_y=True, as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

Фиксируются random_state, что позволяет гарантировать, что в скриптах train_model.py и validate_model.py сплиты данных идентичны.

Пайплайн обучения по шагам:
1. Ждем пока появится файл кофига в директории `configs` (один конфиг - один ран - одна модель)
2. Запускаем скрипт обучения
    1. Для каждого конфига выполняем
        1. Считываем параметры модели и метрики
        2. Строим и обучаем модель
        3. Вычисляем метрики на X_val сплите
        4. Логируем модель, параметры и значения метрик в MLFlow

<p align="center">
  <img width="800" height="300" src="https://github.com/Anteii/ssau-data-engineering-lab-3/blob/main/screenshots/train-model-airflow.png"/>
</p>
<p style="text-align: center">Airflow model training pipline</p>

<p align="center">
  <img width="800" height="300" src="https://github.com/Anteii/ssau-data-engineering-lab-3/blob/main/screenshots/mlflow-experiment-ui.png"/>
</p>
<p style="text-align: center">Эксперимент MLFlow </p>

Проблемы с которыми столкнулся на этом этапе:
1. MLFlow server не мог приконектиться к S3 из-за чего не показывал артифакты эксперимента. Через python API также не получалось их забрать. Решилось через установку boto3 в контейнере с MLFlow server.
 

# Пайплайн валидации моделей

Алгоритм валидации (запуск по таймеру):
1. Для каждего конфига в папке `configs`
    1. Считываем название модели, параметры эксперимента, метрики
    2. Находим эксперимент по имени (полагаем что все модели тестируются в рамках решения одной задачи)
    3. Получаем все раны по experiment-id
    4. Среди них находим ран с параметрами как в конфиге
    5. Получаем артифакт
    6. Вычисляем метрики на X_test сплите и <b>логируем их в тот же ран с постфиксом _test</b>
    7. Находим в списке ранов лучший по таргет метрики с постфиксом _test (если его нет, то выбираем текущий ран)
    8. Модель соответсвующую выбранному рану продвигаем в stage: Production

<p align="center">
  <img width="800" height="300" src="https://github.com/Anteii/ssau-data-engineering-lab-3/blob/main/screenshots/validate-model-airflow.png>
</p>
<p style="text-align: center">Пайплайн валидации Airflow</p>

<p align="center">
  <img width="800" height="300" src="https://github.com/Anteii/ssau-data-engineering-lab-3/blob/main/screenshots/mlflow-stage-ui.png"/>
</p>
<p style="text-align: center">Staging MLFlow </p>

Проблемы с которыми столкнулся на этом этапе:
1. При попыте через mlflow python API забрать артифакт падало с ошибкой bad credentials (как решил не помню)
2. Параметр `artifact_path`. Если `artifact_path = ""`, то артифакты рана логируются в `{run_id}\artifacts`. Если указан, то в `{run_id}\artifacts`. Из-за этого проблемы с получением моделей