import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

output_dataset_location ='/home/ubuntu/exam_PENG/examen-dvc/data/processed_data'
output_scaled_dataset_location ='/home/ubuntu/exam_PENG/examen-dvc/data/scaled'
models_location='/home/ubuntu/exam_PENG/examen-dvc/models'

X_train_scaled = pd.read_csv(output_scaled_dataset_location+'/X_train_scaled.csv')
y_train = pd.read_csv(output_dataset_location+'/y_train.csv')

with open(models_location+'/best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train_scaled, y_train.values.ravel())

with open(models_location+'/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)