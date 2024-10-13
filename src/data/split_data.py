import pandas as pd
from sklearn.model_selection import train_test_split

raw_data_location ='/home/ubuntu/exam_PENG/examen-dvc/data/raw_data/raw.csv'
output_dataset_location ='/home/ubuntu/exam_PENG/examen-dvc/data/processed_data'

data=pd.read_csv(raw_data_location)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv(output_dataset_location+'/X_train.csv', index=False)
X_test.to_csv(output_dataset_location+'/X_test.csv', index=False)
y_train.to_csv(output_dataset_location+'/y_train.csv', index=False)
y_test.to_csv(output_dataset_location+'/y_test.csv', index=False)