import os
import argparse
import pandas as pd
import numpy as np
from evaluate import load_models_and_results, get_brier_curves, plot_brier_curves_with_color_list

def process_folder_brier(base_dir, ignore_svm=False):
    # Define columns for different sets
    cols_22 = ['age', 'education', 'weight', 'height', 'smoke', 'drink', 'no_falls', 'fx50', 'physical',
               'hypertension', 'copd', 'parkinson', 'cancer', 'rheumatoid', 'cvd',
               'renal', 'depression', 'diabetes', 'Tscore', 'protein', 'calcium', 'coffee']
    cols_11 = ['age', 'weight', 'height', 'fx50', 'smoke', 'drink', 'rheumatoid', 'Tscore']
    cols_5 = ['age', 'weight', 'no_falls', 'fx50', 'Tscore']

    # Initialize a dictionary to store the results
    results_dict = {'model': [], '5risks_brier': [], '11risks_brier': [], '22risks_brier': []}

    # Iterate through each folder and compute Brier score curves
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_path}")

            # Determine which column set to use based on folder name
            if '22' in folder:
                cols_x = cols_22
            elif '11' in folder:
                cols_x = cols_11
            elif '5' in folder:
                cols_x = cols_5
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

            # Compute Brier score curves
            brier_curves = get_brier_curves(models_list, train_x, test_x, train_y, test_y, cols_x, times)

            # Plot Brier score curves
            model_name_map = {
                'deepsurv': 'DeepSurv', 'deephit': 'DeepHit',
                'cox_ph': 'CoxPH', 'gboost': 'GradientBoosting',
                'svm': 'SVM', 'rsf': "RSF", 'kaplan_meier': 'KaplanMeier',
                'random': 'Random'
            }
            if ignore_svm:
                brier_curves = brier_curves.drop(columns=['svm'], errors='ignore')
            plot_brier_curves_with_color_list(brier_curves, model_name_map, save_folder=folder_path)

def main():
    parser = argparse.ArgumentParser(description="Compute and plot Brier score curves for models in subfolders.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the base directory.")
    parser.add_argument('--ignore_svm', action='store_true', help="Ignore the SVM model.")
    args = parser.parse_args()

    base_dir = args.folder
    ignore_svm = args.ignore_svm

    process_folder_brier(base_dir, ignore_svm)

if __name__ == "__main__":
    main()
