import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder
import os

# Model configuration
model_name = "trl-lib/Qwen2-0.5B-Reward"
repo_path = "NotTheStallion/Qwen2-Reward-Model-0.5B"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Original model architecture:")
print(model)


# Modify the model's head
if hasattr(model, "lm_head"):  
    in_features = model.lm_head.in_features
    model.lm_head = nn.Sequential(
        nn.Linear(in_features, 1), 
        nn.Sigmoid()  # Changed from Softmax to Sigmoid for reward modeling
    )
else:
    raise AttributeError("The model does not have a 'lm_head' attribute to modify.")

print("\nModified model architecture:")
print(model)

# Authenticate with Hugging Face Hub
api = HfApi()
token = HfFolder.get_token()
if not token:
    raise ValueError("No Hugging Face token found. Please log in using `huggingface-cli login`.")

# Create or clear the repository
try:
    api.delete_repo(repo_path, token=token)
    print(f"\nRepository '{repo_path}' deleted successfully.")
except Exception as e:
    print(f"\nRepository '{repo_path}' doesn't exist or couldn't be deleted: {e}")

# Create a new repository
create_repo(repo_path, token=token, private=True, exist_ok=True)
print(f"\nRepository '{repo_path}' created successfully.")

# Save model and tokenizer to local directory
local_dir = "./qwen2_reward_model"
os.makedirs(local_dir, exist_ok=True)

model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)

# Upload to Hugging Face Hub
upload_folder(
    folder_path=local_dir,
    repo_id=repo_path,
    repo_type="model",
    commit_message="Initial commit of modified Qwen2 reward model",
)

print("\nModified model successfully pushed to Hugging Face Hub!")