import mlflow
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="sk-learn-random-forest-reg-model", version=3, stage="Production"
)