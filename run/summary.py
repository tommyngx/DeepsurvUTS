import os
import pandas as pd
import argparse
from utils import get_csv_files, plot_performance_benchmark

def main():
    parser = argparse.ArgumentParser(description="Retrieve and print CSV files from subfolders.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the base directory.")
    parser.add_argument('--keyword', type=str, required=True, help="Keywords to select folders (e.g., 'SOF_anyfx').")
    args = parser.parse_args()

    base_dir = args.folder
    keywords = args.keyword.split('_')
    summary_dir = os.path.join(base_dir, 'summary')

    merged_df = get_csv_files(base_dir, keywords)
    if not merged_df.empty:
        print(merged_df)
        plot_performance_benchmark(merged_df, summary_dir)
    else:
        print("No matching CSV files found.")

if __name__ == "__main__":
    main()
