import pandas as pd
from sklearn.preprocessing import StandardScaler

input_dataset_location ='/home/ubuntu/exam_PENG/examen-dvc/data/processed_data'
output_dataset_location ='/home/ubuntu/exam_PENG/examen-dvc/data/scaled'

X_train = pd.read_csv(input_dataset_location+'/X_train.csv')
X_test = pd.read_csv(input_dataset_location+'/X_test.csv')

X_train_numeric = X_train.select_dtypes(include=['float64', 'int64'])
X_test_numeric = X_test.select_dtypes(include=['float64', 'int64'])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)

pd.DataFrame(X_train_scaled, columns=X_train_numeric.columns).to_csv(output_dataset_location+'/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test_numeric.columns).to_csv(output_dataset_location+'/X_test_scaled.csv', index=False)