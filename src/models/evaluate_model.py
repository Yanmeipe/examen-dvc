import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json

output_dataset_location ='/home/ubuntu/exam_PENG/examen-dvc/data/processed_data'
output_scaled_dataset_location ='/home/ubuntu/exam_PENG/examen-dvc/data/scaled'
models_location='/home/ubuntu/exam_PENG/examen-dvc/models'
prediction_data_location='/home/ubuntu/exam_PENG/examen-dvc/data/prediction'
metrics_location='/home/ubuntu/exam_PENG/examen-dvc/metrics'

X_test_scaled = pd.read_csv(output_scaled_dataset_location+'/X_test_scaled.csv')
y_test = pd.read_csv(output_dataset_location+'/y_test.csv')

with open(models_location+'/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {
    'mse': mse,
    'r2': r2
}

pd.DataFrame(y_pred, columns=['predictions']).to_csv(prediction_data_location+'/predictions.csv', index=False)

with open(metrics_location+'/scores.json', 'w') as f:
    json.dump(metrics, f)

