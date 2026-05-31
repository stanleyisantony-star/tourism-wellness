
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split # Import for stratified split
from huggingface_hub import HfApi

hf_username = 'HfStan'
dataset_repo_id = f"{hf_username}/tourism-wellness-dataset"

base_dir = 'tourism_project'
data_dir = os.path.join(base_dir, 'data')
csv_path = os.path.join(data_dir, 'tourism.csv')

# Ensure the project data directory exists
os.makedirs(data_dir, exist_ok=True)

try:
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded {csv_path}. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: {csv_path} not found. Please ensure tourism.csv is in the correct path.")
    exit(1)

# Perform a stratified train-test split for registration
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ProdTaken'])

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))
ds_dict = DatasetDict({'train': train_ds, 'test': test_ds})

api = HfApi(token=os.environ.get('HF_TOKEN'))
api.create_repo(repo_id=dataset_repo_id, repo_type='dataset', exist_ok=True)
ds_dict.push_to_hub(dataset_repo_id, token=os.environ.get('HF_TOKEN'))

print('Dataset pushed to:', dataset_repo_id)
