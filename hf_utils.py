from huggingface_hub import HfApi
import os

def ensure_hf_repo_exists(repo_name, local_dir, token=None):
    """
    Ensures a Hugging Face repository exists. If it exists, it overwrites it with the new directory.
    If it doesn't exist, it creates the repository.

    Args:
        repo_name (str): Name of the Hugging Face repository (e.g., "username/repo_name").
        local_dir (str): Path to the local directory to sync with the repository.
        token (str): Hugging Face authentication token.
    """
    api = HfApi()

    try:
        # Check if the repository exists
        api.repo_info(repo_name, token=token)
        print(f"Repository '{repo_name}' exists. Overwriting with the new directory...")
    except :
        # Create the repository if it doesn't exist
        print(f"Repository '{repo_name}' does not exist. Creating it...")
        api.create_repo(repo_name, token=token)


    # Push the local directory to the Hugging Face repository
    print(f"Overwriting the repository '{repo_name}' on Hugging Face with the contents of '{local_dir}'...")
    api.upload_folder(folder_path=local_dir, repo_id=repo_name, token=token)
    print(f"Repository '{repo_name}' has been successfully updated.")
    
    
if __name__ == "__main__":
    # Example usage
    repo_name = "NotTheStallion/llama3-3.2-1b-RewardModel"  # Replace with your repo name
    local_dir = "../llama3-3.2-1b-rm"  # Replace with your local directory

    ensure_hf_repo_exists(repo_name, local_dir)