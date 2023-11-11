FROM python:3.10-slim
# Set where tracing should be sended
ENV MLFLOW_TRACKING_URI=http://mlflow_server:5000
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ENV AWS_ACCESS_KEY_ID=minio
ENV AWS_SECRET_ACCESS_KEY=minio123
# Set timeout otherwise long build will crash in pip
RUN apt update && apt install -y git
RUN pip install --default-timeout=100 scikit-learn mlflow boto3