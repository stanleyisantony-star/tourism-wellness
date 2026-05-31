
import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

hf_username = 'HfStan'
raw_dataset_repo_id = f"{hf_username}/tourism-wellness-dataset"
processed_dataset_repo_id = f"{hf_username}/tourism-wellness-processed-dataset"

base_dir = 'tourism_project'
data_dir = os.path.join(base_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

print(f"Loading raw dataset from {raw_dataset_repo_id}...")
raw_dataset = load_dataset(raw_dataset_repo_id, token=os.environ.get('HF_TOKEN'))
df = pd.concat([raw_dataset['train'].to_pandas(), raw_dataset['test'].to_pandas()], ignore_index=True)

print('Shape of raw data from Hub:', df.shape)

df['Gender'] = df['Gender'].str.strip()
df['Occupation'] = df['Occupation'].str.strip()
df['MaritalStatus'] = df['MaritalStatus'].str.strip()

if 'CustomerID' in df.columns:
    df = df.drop(columns=['CustomerID'])
    print('Dropped CustomerID column')

df = df.dropna(how='all')
df = df.ffill()
print('Shape after cleaning:', df.shape)

target_col = 'ProdTaken'
if target_col not in df.columns:
    print(f"Error: Target column '{target_col}' not found in DataFrame.")
    exit(1)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
print('Train shape (processed):', train_df.shape)
print('Test shape (processed):', test_df.shape)

train_hf_processed = Dataset.from_pandas(train_df.reset_index(drop=True))
test_hf_processed = Dataset.from_pandas(test_df.reset_index(drop=True))
ds_dict_processed = DatasetDict({'train': train_hf_processed, 'test': test_hf_processed})

api = HfApi(token=os.environ.get('HF_TOKEN'))
api.create_repo(repo_id=processed_dataset_repo_id, repo_type='dataset', exist_ok=True)
ds_dict_processed.push_to_hub(processed_dataset_repo_id, token=os.environ.get('HF_TOKEN'))

print('Processed DatasetDict pushed to HF Hub as:', processed_dataset_repo_id)
