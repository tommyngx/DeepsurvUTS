import os
import argparse
import pandas as pd
import numpy as np
from utils import loading_config
from evaluate import load_models_and_results, get_brier_curves
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def plot_brier_curves_with_color_list(brier_curves, model_name_map=None, color_list=None, font_prop=None, save_folder=None):
    plt.figure(figsize=(8, 6))

    # Ensure the number of models does not exceed the color list length
    models = [m for m in brier_curves.columns if m != 'time']

    # Plot each column (except 'time') against the time column with assigned colors
    for idx, m in enumerate(models):
        color = color_list[idx % len(color_list)]  # Assign color from the list

        # Use model_name_map if provided, otherwise use original names
        display_name = model_name_map.get(m, m) if model_name_map else m

        # Plot the curve and scatter points
        plt.plot(brier_curves['time'], brier_curves[m] * 100, label=display_name, linestyle='-', color=color)
        plt.scatter(brier_curves['time'], brier_curves[m] * 100, marker='o', s=20, color=color)

    # Customize the plot
    plt.title("Brier Score Curves",  fontproperties=font_prop, fontsize=16, pad=10)
    plt.xlabel("Time (years)", fontsize=14, fontproperties=font_prop)
    plt.ylabel("Brier Score (%)", fontsize=14, fontproperties=font_prop)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(decimals=0))  # Format y-axis as percentages without decimals

    # Customize legend with a white background
    legend = plt.legend(prop=font_prop, fontsize=13)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    # Add lines and markers to the legend
    for handle in legend.legend_handles:
        handle.set_linestyle('-')
        handle.set_marker('o')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot if save_folder is provided
    if save_folder:
        save_path = f"{save_folder}/brier_curves.png"
        plt.savefig(save_path, format='png')

    plt.show()

def process_folder_brier(base_dir, keywords, ignore_svm=True):
    # Load configuration
    config, font_prop, model_name_map, color_list, cols_22, cols_11, cols_5 = loading_config()

    # Initialize a dictionary to store the results
    results_dict = {'model': [], '5risks_brier': [], '11risks_brier': [], '22risks_brier': []}

    # Iterate through each folder and compute Brier score curves
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and all(keyword in folder for keyword in keywords):
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
            if ignore_svm:
                brier_curves = brier_curves.drop(columns=['svm'], errors='ignore')
            
            # Sort models alphabetically
            brier_curves = brier_curves[['time'] + sorted(brier_curves.columns.drop('time'))]

            # Ensure the summary directory exists
            summary_dir = os.path.join(base_dir, 'summary')
            os.makedirs(summary_dir, exist_ok=True)
            
            plot_brier_curves_with_color_list(brier_curves, model_name_map, color_list, font_prop, save_folder=summary_dir)

def main():
    parser = argparse.ArgumentParser(description="Compute and plot Brier score curves for models in subfolders.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the base directory.")
    parser.add_argument('--keyword', type=str, required=True, help="Keywords to select folders (e.g., 'SOF_anyfx').")
    parser.add_argument('--ignore_svm', action='store_true', default=True, help="Ignore the SVM model.")
    args = parser.parse_args()

    base_dir = args.folder
    keywords = args.keyword.split('_')
    ignore_svm = args.ignore_svm

    process_folder_brier(base_dir, keywords, ignore_svm)

if __name__ == "__main__":
    main()
