import os
import pandas as pd
import argparse
from utils import get_csv_files, plot_performance_benchmark
from evaluate import load_models_and_results, get_integrated_brier_score
import numpy as np

def process_folders_brier(base_dir, keywords, summary_dir, results_dict):
    # Define columns for different sets
    cols_22 = ['age', 'education', 'weight', 'height', 'smoke', 'drink', 'no_falls', 'fx50', 'physical',
               'hypertension', 'copd', 'parkinson', 'cancer', 'rheumatoid', 'cvd',
               'renal', 'depression', 'diabetes', 'Tscore', 'protein', 'calcium', 'coffee']
    cols_11 = ['age', 'weight', 'height', 'fx50', 'smoke', 'drink', 'rheumatoid', 'Tscore' ] #'MedYes'
    cols_5 = ['age', 'weight', 'no_falls', 'fx50', 'Tscore']

    # Iterate through each folder and compute integrated Brier scores
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and all(keyword in folder for keyword in keywords):
            print(f"Processing folder: {folder_path}")

            # Determine which column set to use based on folder name
            if '22' in folder:
                cols_x = cols_22
                risk_key = '22risks_brier'
            elif '11' in folder:
                cols_x = cols_11
                risk_key = '11risks_brier'
            elif '5' in folder:
                cols_x = cols_5
                risk_key = '5risks_brier'
            else:
                raise ValueError("Folder name must contain '22', '11', or '5'.")

            # Load models and results
            models_list, results_table = load_models_and_results(folder_path, cols_x)

            # Define time points for evaluation
            times = np.arange(1, 20)

            # Load train and test data
            train_x = pd.read_pickle(os.path.join(folder_path, 'data', 'train_x.pkl'))
            test_x = pd.read_pickle(os.path.join(folder_path, 'data', 'test_x.pkl'))
            train_y = pd.read_pickle(os.path.join(folder_path, 'data', 'train_y.pkl'))
            test_y = pd.read_pickle(os.path.join(folder_path, 'data', 'test_y.pkl'))

            # Compute integrated Brier scores
            integrated_scores = get_integrated_brier_score(models_list, train_x, test_x, train_y, test_y, cols_x, times)
            #print(f"Integrated Brier Scores for {folder_path}: {integrated_scores}")

            # Filter out 'kaplan_meier' and 'random' and store the results
            filtered_scores = {k: v for k, v in integrated_scores.items() if k not in ['kaplan_meier', 'random']}
            for model, score in filtered_scores.items():
                if model not in results_dict['model']:
                    results_dict['model'].append(model)
                    results_dict['5risks_brier'].append(None)
                    results_dict['11risks_brier'].append(None)
                    results_dict['22risks_brier'].append(None)
                model_index = results_dict['model'].index(model)
                results_dict[risk_key][model_index] = score

    # Create a DataFrame from the results dictionary
    results_df = pd.DataFrame(results_dict).dropna(axis=1, how='all')
    #print(results_df)

def plot_performance_benchmark(df, summary_dir):
    import matplotlib.pyplot as plt

    # Ensure the summary directory exists
    os.makedirs(summary_dir, exist_ok=True)

    # Plot cindex vs Brier scores
    plt.figure(figsize=(10, 6))
    for model in df['model']:
        plt.scatter(df[df['model'] == model]['5risks_brier'], df[df['model'] == model]['5risks_cindex'], label=f'{model} (5 risks)')
        plt.scatter(df[df['model'] == model]['11risks_brier'], df[df['model'] == model]['11risks_cindex'], label=f'{model} (11 risks)')
        plt.scatter(df[df['model'] == model]['22risks_brier'], df[df['model'] == model]['22risks_cindex'], label=f'{model} (22 risks)')

    plt.xlabel('Brier Score')
    plt.ylabel('C-index')
    plt.title('Performance Benchmark: C-index vs Brier Score')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(summary_dir, 'performance_benchmark.png')
    plt.savefig(plot_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Retrieve and print CSV files from subfolders.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the base directory.")
    parser.add_argument('--keyword', type=str, required=True, help="Keywords to select folders (e.g., 'SOF_anyfx').")
    args = parser.parse_args()

    base_dir = args.folder
    keywords = args.keyword.split('_')
    summary_dir = os.path.join(base_dir, 'summary')

    # Initialize a dictionary to store the results
    results_dict = {'model': [], '5risks_brier': [], '11risks_brier': [], '22risks_brier': []}

    # Get cindex from CSV files
    cindex_df = get_csv_files(base_dir, keywords)
    if not cindex_df.empty:
        # Rename columns for cindex
        cindex_df = cindex_df.rename(columns={'5 risks': '5risks_cindex', '11 risks': '11risks_cindex', '22 risks': '22risks_cindex'})
        #print(cindex_df)

        # Process folders to get Brier scores
        process_folders_brier(base_dir, keywords, summary_dir, results_dict)

        # Get Brier scores DataFrame
        brier_df = pd.DataFrame(results_dict).dropna(axis=1, how='all')

        # Merge cindex and Brier scores DataFrames on 'model'
        final_df = pd.merge(cindex_df, brier_df, on='model')
        print(final_df)

        # Plot performance benchmark
        plot_performance_benchmark(final_df, summary_dir)
    else:
        print("No matching CSV files found.")

if __name__ == "__main__":
    main()
