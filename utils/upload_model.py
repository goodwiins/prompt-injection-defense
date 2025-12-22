
from huggingface_hub import HfApi, create_repo
import os

def upload_model():
    # Get user input for Hugging Face username and repo name
    hf_username = input("Enter your Hugging Face username: ")
    repo_name = input("Enter the desired repository name for your model: ")

    # Create the repository on the Hugging Face Hub
    repo_url = create_repo(repo_id=f"{hf_username}/{repo_name}", exist_ok=True)
    print(f"Repository created: {repo_url}")

    # Upload the files
    api = HfApi()
    api.upload_folder(
        folder_path="injection_aware_mpnet_model",
        repo_id=f"{hf_username}/{repo_name}",
        repo_type="model",
    )
    print("Files uploaded successfully!")

if __name__ == "__main__":
    upload_model()
