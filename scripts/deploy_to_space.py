
import os
import shutil
from huggingface_hub import HfApi, create_repo
from datasets import load_dataset
import joblib
import pandas as pd

hf_username = 'HfStan'
model_repo_id = f"{hf_username}/tourism-wellness-model"
space_repo_id = f"{hf_username}/tourism-wellness-space"

base_dir = 'tourism_project'
deployment_dir = os.path.join(base_dir, 'deployment')
os.makedirs(deployment_dir, exist_ok=True)

app_path = os.path.join(deployment_dir, 'app.py')
requirements_path = os.path.join(deployment_dir, 'requirements.txt')
dockerfile_path = os.path.join(deployment_dir, 'Dockerfile')

api = HfApi(token=os.environ.get('HF_TOKEN'))
print(f"Downloading best_model.joblib from {model_repo_id}...")
api.hf_hub_download(
    repo_id=model_repo_id,
    filename='best_model.joblib',
    local_dir=deployment_dir,
    repo_type='model',
    token=os.environ.get('HF_TOKEN')
)
print(f"best_model.joblib downloaded to {deployment_dir}")

try:
    create_repo(repo_id=space_repo_id, repo_type='space', exist_ok=True, space_sdk='docker', token=os.environ.get('HF_TOKEN'))
    print(f"Hugging Face Space '{space_repo_id}' ensured to exist.")
except Exception as e:
    print(f"Could not create/verify Hugging Face Space: {e}")
    print("Please ensure your HF_TOKEN has write permissions for user spaces and space_sdk is correct.")
    exit(1)

files_to_upload = ['app.py', 'Dockerfile', 'requirements.txt', 'best_model.joblib']

for file_name in files_to_upload:
    local_file_path = os.path.join(deployment_dir, file_name)
    if os.path.exists(local_file_path):
        api.upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo=file_name,
            repo_id=space_repo_id,
            repo_type='space',
            commit_message=f'Update {file_name} for Streamlit Space'
        )
        print(f'{file_name} uploaded to Hugging Face Space successfully!')
    else:
        print(f"Error: {local_file_path} not found. Skipping upload for {file_name}.")
        exit(1)

print('Deployment to Hugging Face Space complete.')
