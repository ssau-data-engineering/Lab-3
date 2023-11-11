import argparse
import glob
import os
import json
import datetime

from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score

from mlflow.models import infer_signature
import mlflow

AVAILABLE_CLASSIFIERS_NAME2MODEL_DICT = {
    'KNeighborsClassifier': KNeighborsClassifier,
    'SVC': SVC,
    'GaussianProcessClassifier': GaussianProcessClassifier,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'MLPClassifier': MLPClassifier,
    'AdaBoostClassifier': AdaBoostClassifier,
    'GaussianNB': GaussianNB,
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis,
}

def train(config):
    with open(config, 'r') as jr:
        config_data = json.load(jr)

    if AVAILABLE_CLASSIFIERS_NAME2MODEL_DICT.get(config_data['model']) is None:
        raise Exception('Unknown model name')

    with mlflow.start_run():
        digits = load_digits()
        # flatten the images
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))
        # Split data into 50% train and 50% test subsets
        X_train, X_test, y_train, y_test = train_test_split(
            data, digits.target, test_size=0.5, shuffle=False
        )


        clf = AVAILABLE_CLASSIFIERS_NAME2MODEL_DICT.get(config_data['model'])(**config_data['parameters'])
        mlflow.log_params(config_data)
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted = clf.predict(X_test)

        f1_w = f1_score(y_test, predicted, average='weighted') 
        f1_avg = f1_score(y_test, predicted, average='macro') 
        acc = accuracy_score(y_test, predicted)
        mlflow.log_metric("f1_w", f1_w)
        mlflow.log_metric("f1_avg", f1_avg)
        mlflow.log_metric("acc", acc)
        
        signature = infer_signature(X_test, predicted)
        mlflow.sklearn.log_model(
            clf, 'sk_models', 
            signature=signature, 
            # Name must be unique, so just add datetime
            registered_model_name='sklearn-model-%s-%s' % (config_data['model'], datetime.datetime.now()),
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train sklearn model on config file.')
    parser.add_argument('configs_folder_path', type=str,
                        help='Path to folder with configs of the sklearn models.')

    args = parser.parse_args()
    configs_files_path = glob.glob(os.path.join(args.configs_folder_path, '*.json'))
    for config_file_path in configs_files_path:
        train(config_file_path)

                                  