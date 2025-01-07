import os
import pandas as pd
import argparse
from utils import get_csv_files, plot_performance_benchmark
from evaluate import load_models_and_results, get_integrated_brier_score
import numpy as np

def process_folders(base_dir, keywords, summary_dir):
    # Define columns for different sets
    cols_22 = ['age', 'education', 'weight', 'height', 'smoke', 'drink', 'no_falls', 'fx50', 'physical',
               'hypertension', 'copd', 'parkinson', 'cancer', 'rheumatoid', 'cvd',
               'renal', 'depression', 'diabetes', 'Tscore', 'protein', 'calcium', 'coffee']
    cols_11 = ['age', 'weight', 'height', 'fx50', 'smoke', 'drink', 'rheumatoid', 'Tscore', 'MedYes']
    cols_5 = ['age', 'weight', 'no_falls', 'fx50', 'Tscore']

    # Iterate through each folder and compute integrated Brier scores
    for root, dirs, files in os.walk(base_dir):
        if all(keyword in root for keyword in keywords):
            path_dir = root
            print(f"Processing folder: {path_dir}")

            # Determine which column set to use based on folder name
            if '22' in path_dir:
                cols_x = cols_22
            elif '11' in path_dir:
                cols_x = cols_11
            elif '5' in path_dir:
                cols_x = cols_5
            else:
                raise ValueError("Folder name must contain '22', '11', or '5'.")

            # Load models and results
            models_list, results_table = load_models_and_results(path_dir, cols_x)

            # Define time points for evaluation
            times = np.arange(1, 20)

            # Load train and test data
            train_x = pd.read_pickle(os.path.join(path_dir, 'data', 'train_x.pkl'))
            test_x = pd.read_pickle(os.path.join(path_dir, 'data', 'test_x.pkl'))
            train_y = pd.read_pickle(os.path.join(path_dir, 'data', 'train_y.pkl'))
            test_y = pd.read_pickle(os.path.join(path_dir, 'data', 'test_y.pkl'))

            # Compute integrated Brier scores
            integrated_scores = get_integrated_brier_score(models_list, train_x, test_x, train_y, test_y, cols_x, times)
            print(f"Integrated Brier Scores for {path_dir}: {integrated_scores}")

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
        process_folders(base_dir, keywords, summary_dir)
    else:
        print("No matching CSV files found.")

if __name__ == "__main__":
    main()
