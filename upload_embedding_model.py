
from huggingface_hub import HfApi, create_repo
import os

def upload_embedding_model():
    # Get user input for Hugging Face username and repo name
    hf_username = "goodwiinz"
    repo_name = "all-mpnet-base-v2"

    # Create the repository on the Hugging Face Hub
    repo_url = create_repo(repo_id=f"{hf_username}/{repo_name}", exist_ok=True)
    print(f"Repository ensured to exist: {repo_url}")

    # Upload the files
    api = HfApi()
    api.upload_folder(
        folder_path="models/injection_aware_mpnet",
        repo_id=f"{hf_username}/{repo_name}",
        repo_type="model",
    )
    print("Files uploaded successfully!")

if __name__ == "__main__":
    upload_embedding_model()
