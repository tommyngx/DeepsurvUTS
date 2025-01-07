import os
import pandas as pd
import argparse

def get_csv_files(base_dir, keywords):
    """
    Retrieve CSV files from subfolders that match the specified keywords.

    Args:
        base_dir (str): Base directory to search.
        keywords (list): List of keywords to match folder names.

    Returns:
        list: List of DataFrames from the matched CSV files.
    """
    dataframes = []
    for root, dirs, files in os.walk(base_dir):
        if all(keyword in root for keyword in keywords):
            for file in files:
                if file == 'result.csv':
                    csv_path = os.path.join(root, file)
                    df = pd.read_csv(csv_path)
                    dataframes.append(df)
                    print(f"Loaded DataFrame from {csv_path}")
    return dataframes

def main():
    parser = argparse.ArgumentParser(description="Retrieve and print CSV files from subfolders.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the base directory.")
    parser.add_argument('--keyword', type=str, required=True, help="Keywords to select folders (e.g., 'SOF_anyfx').")
    args = parser.parse_args()

    base_dir = args.folder
    keywords = args.keyword.split('_')

    dataframes = get_csv_files(base_dir, keywords)
    for df in dataframes:
        print(df)

if __name__ == "__main__":
    main()
