
import os
import pandas as pd
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from huggingface_hub import HfApi

hf_username = 'HfStan'
processed_dataset_repo_id = f"{hf_username}/tourism-wellness-processed-dataset"
model_repo_id = f"{hf_username}/tourism-wellness-model"

base_dir = 'tourism_project'
models_dir = os.path.join(base_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

print(f"Loading processed dataset from {processed_dataset_repo_id}...")
processed_dataset = load_dataset(processed_dataset_repo_id, token=os.environ.get('HF_TOKEN'))
train_df = processed_dataset['train'].to_pandas()
test_df = processed_dataset['test'].to_pandas()

print('Train shape (loaded from Hub):', train_df.shape)
print('Test shape (loaded from Hub):', test_df.shape)

target_col = 'ProdTaken'
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

param_grid = {
    'model__n_estimators': [100, 200],
    ''model__max_depth': [None, 10, 20]
}
best_f1 = -np.inf
best_params = None
best_model = None

mlflow.set_experiment('tourism_wellness_prediction')

for n_estimators in param_grid['model__n_estimators']:
    for max_depth in param_grid['model__max_depth']:
        with mlflow.start_run():

            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            model = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', clf)])
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_param('n_estimators', n_estimators)
            mlflow.log_param('max_depth', max_depth)
            mlflow.log_metric('accuracy', acc)
            mlflow.log_metric('f1_score', f1)
            print(f'Run with n_estimators={n_estimators}, max_depth={max_depth}, accuracy={acc:.4f}, f1={f1:.4f}')

            if f1 > best_f1:
                best_f1 = f1
                best_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
                best_model = model

print('Best F1 score:', best_f1)
print('Best params:', best_params)

best_model_path = os.path.join(models_dir, 'best_model.joblib')
joblib.dump(best_model, best_model_path)
print('Best model saved to', best_model_path)

api = HfApi(token=os.environ.get('HF_TOKEN'))
api.create_repo(repo_id=model_repo_id, repo_type='model', exist_ok=True)
api.upload_file(
    path_or_fileobj=best_model_path,
    path_in_repo='best_model.joblib',
    repo_id=model_repo_id,
    repo_type='model',
    commit_message='Add best tourism wellness prediction model from CI/CD'
)
print('Model uploaded to Hugging Face Hub successfully!')
