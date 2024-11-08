import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle

output_dataset_location ='/home/ubuntu/exam_PENG/examen-dvc/data/processed_data'
output_scaled_dataset_location ='/home/ubuntu/exam_PENG/examen-dvc/data/scaled'
models_location='/home/ubuntu/exam_PENG/examen-dvc/models'

X_train_scaled = pd.read_csv(output_scaled_dataset_location+'/X_train_scaled.csv')
y_train = pd.read_csv(output_dataset_location+'/y_train.csv')

model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train.values.ravel())

with open(models_location+'/best_params.pkl', 'wb') as f:
    pickle.dump(grid_search.best_params_, f)