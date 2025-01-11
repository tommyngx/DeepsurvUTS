import os
import argparse
import subprocess
import sys

def check_required_libraries():
    """Check if all required libraries are available."""
    required_libs = {
        'os': 'Built-in',
        'argparse': 'Built-in',
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
        sys.exit(1)
    else:
        print("\nAll required libraries are available!")

def update_repo(folder_path, github_acc, github_email, commit_message, branch="main"):
    """
    Update GitHub repository from existing folder.

    Args:
        folder_path (str): Path to the Git repository folder.
        github_acc (str): GitHub account username.
        github_email (str): GitHub email for configuration.
        commit_message (str): Commit message for the update.
        branch (str): Branch to push changes to. Default is "main".
    """
    try:
        # Change to repository directory
        os.chdir(folder_path)
        print(f"Changed to directory: {folder_path}")
        
        # Configure Git
        print("Configuring Git...")
        subprocess.run(["git", "config", "user.name", github_acc], check=True)
        subprocess.run(["git", "config", "user.email", github_email], check=True)

        # Add all changes
        print("Adding changes...")
        subprocess.run(["git", "add", "."], check=True)
        
        # Commit changes
        print("Committing changes...")
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        
        # Push changes
        print("Pushing changes...")
        subprocess.run(["git", "push", "origin", branch], check=True)
        
        print(f"Changes pushed successfully to the {branch} branch!")
        
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    check_required_libraries()
    
    parser = argparse.ArgumentParser(description="Update existing Git repository to GitHub.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the Git repository folder.")
    parser.add_argument('--acc', type=str, default='Osteolab', help="GitHub account username.")
    parser.add_argument('--email', type=str, default="tommylimitless@gmail.com", help="GitHub email.")
    parser.add_argument('--message', type=str, required=True, help="Commit message for the update.")
    parser.add_argument('--branch', type=str, default="main", help="Branch to push changes to.")
    
    args = parser.parse_args()
    
    try:
        update_repo(
            folder_path=args.folder,
            github_acc=args.acc,
            github_email=args.email,
            commit_message=args.message,
            branch=args.branch
        )
    except Exception as e:
        print(f"Error during update: {e}")

if __name__ == "__main__":
    main()
