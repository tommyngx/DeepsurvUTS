import os
import argparse
import shutil
import subprocess
import sys

def copy_folder_contents(source_folder, destination_folder):
    """
    Copy source_folder as a subfolder inside destination_folder, while retaining existing items in destination_folder.
    
    Args:
        source_folder (str): Path to the source folder.
        destination_folder (str): Path to the destination folder.
    """
    try:
        # Ensure source exists
        if not os.path.exists(source_folder):
            raise FileNotFoundError(f"Source folder '{source_folder}' does not exist.")
        
        # Ensure destination exists
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)  # Create destination folder if it doesn't exist
        
        # Define the new path for source_folder inside destination_folder
        new_subfolder = os.path.join(destination_folder, os.path.basename(source_folder))
        
        # Copy source_folder into destination_folder as a subfolder
        shutil.copytree(source_folder, new_subfolder, dirs_exist_ok=True)  # Merge if subfolder exists
        
        print(f"'{source_folder}' copied successfully as '{new_subfolder}' inside '{destination_folder}'")
    except Exception as e:
        print(f"Error: {e}")

def check_required_libraries():
    """Check if all required libraries are available."""
    required_libs = {
        'os': 'Built-in',
        'argparse': 'Built-in',
        'shutil': 'Built-in',
        'subprocess': 'Built-in',
        'sys': 'Built-in'
    }
    
    missing_libs = []
    for lib, source in required_libs.items():
        try:
            __import__(lib)
            print(f"✓ {lib} (from {source})")
        except ImportError:
            missing_libs.append(lib)
            print(f"✗ {lib} not found")
    
    if missing_libs:
        print("\nMissing libraries:", ", ".join(missing_libs))
        print("Please install missing libraries using pip:")
        print(f"pip install {' '.join(missing_libs)}")
        sys.exit(1)
    else:
        print("\nAll required libraries are available!")

def clone_and_update_repo(github_acc="Osteolab", github_email="tommylimitless@gmail.com", github_key="X", 
                          folderX="/path/to/source", commit_message="Update", branch="main"):
    """
    Clone a GitHub repository, commit changes, and push updates.

    Args:
        github_acc (str): GitHub account username.
        github_email (str): GitHub email for configuration.
        github_key (str): GitHub personal access token.
        folderX (str): Path to the folder whose contents will be added to the repo.
        commit_message (str): Commit message for the update.
        branch (str): Branch to push changes to. Default is "main".
    """
    try:
        # Define the repository name and paths
        folder_name = "DeepsurvUTS_results"
        repo_path = f"/content/{folder_name}"
        
        # Step 1: Remove existing folder if it exists
        if os.path.exists(repo_path):
            print(f"Removing existing folder: {repo_path}")
            shutil.rmtree(repo_path)
        
        # Step 2: Clone the repository
        repo_url = f"https://{github_acc}:{github_key}@github.com/tommyngx/{folder_name}"
        print(f"Cloning repository from: {repo_url}")
        os.chdir("/content")
        subprocess.run(["git", "clone", repo_url], check=True)
        
        # Step 3: Configure Git
        print("Configuring Git...")
        subprocess.run(["git", "config", "--global", "user.name", github_acc], check=True)
        subprocess.run(["git", "config", "--global", "user.email", github_email], check=True)

        # Step 4: Check if folderX name exists in the destination and remove it
        folderX_name = os.path.basename(folderX)  # Extract the folder name
        destination_subfolder = os.path.join(repo_path, folderX_name)

        if os.path.exists(destination_subfolder):
            print(f"Removing existing folder: {destination_subfolder}")
            shutil.rmtree(destination_subfolder)

        # Copy contents from folderX to the cloned repository
        print(f"Copying contents from {folderX} to {repo_path}...")
        copy_folder_contents(folderX, repo_path)

        # Cleanup specific files (optional, based on use case)
        #for item in ['data', 'models', 'model_scores_dl2.csv', 'outputs_mros']:
        for item in ['data', 'models', 'model_scores_dl2.csv', 'outputs_sof', 'outputs_mros']:
            item_path = os.path.join(repo_path, item)
            if os.path.exists(item_path):
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        
        # Step 5: Add, commit, and push changes
        os.chdir(repo_path)
        print("Adding, committing, and pushing changes...")
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=False)
        subprocess.run(["git", "push", "origin", branch], check=True)
        
        print(f"Changes pushed successfully to the {branch} branch!")
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    # Check required libraries first
    check_required_libraries()
    
    parser = argparse.ArgumentParser(description="Update results to GitHub repository.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the folder to update.")
    parser.add_argument('--acc', type=str, default='Osteolab', help="GitHub account username.")
    parser.add_argument('--email', type=str, default="tommylimitless@gmail.com", help="GitHub email.")
    parser.add_argument('--key', type=str, required=True, help="GitHub personal access token.")
    parser.add_argument('--message', type=str, required=True, help="Commit message for the update.")
    parser.add_argument('--branch', type=str, default="main", help="Branch to push changes to.")
    
    args = parser.parse_args()
    
    try:
        clone_and_update_repo(
            github_acc=args.acc,
            github_email=args.email,
            github_key=args.key,
            folderX=args.folder,
            commit_message=args.message,
            branch=args.branch
        )
        print("Update completed successfully!")
    except Exception as e:
        print(f"Error during update: {e}")

if __name__ == "__main__":
    main()
