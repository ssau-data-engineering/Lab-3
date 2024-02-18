from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

digits = load_digits()

data = {}

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, shuffle=True, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=True, random_state=42)

data['x_train'] = x_train
data['y_train'] = y_train
data['x_val'] = x_val
data['y_val'] = y_val
data['x_test'] = x_test
data['y_test'] = y_test

mean_image = np.mean(x_train, axis=0)
print(mean_image.shape)

x_train -= mean_image
x_val -= mean_image
x_test -= mean_image

x_train_csv = pd.DataFrame(x_train)
x_train_csv.to_csv('/opt/airflow/data/x_train.csv', index=False)
y_train_csv = pd.DataFrame(y_train)
y_train_csv.to_csv('/opt/airflow/data/y_train.csv', index=False)

x_val_csv = pd.DataFrame(x_val)
x_val_csv.to_csv('/opt/airflow/data/x_val.csv', index=False)
y_val_csv = pd.DataFrame(y_val)
y_val_csv.to_csv('/opt/airflow/data/y_val.csv', index=False)

x_test_csv = pd.DataFrame(x_test)
x_test_csv.to_csv('/opt/airflow/data/x_test.csv', index=False)
y_test_csv = pd.DataFrame(y_test)
y_test_csv.to_csv('/opt/airflow/data/y_test.csv', index=False)
