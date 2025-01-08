import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils import loading_config
from sklearn.calibration import calibration_curve
from evaluate import load_models_and_results, generate_all_probabilities


# Load configuration
config, font_prop, model_name_map, color_list, cols_22, cols_11, cols_5 = loading_config()

def plot_10_year_calibration_curve(models_to_plot, all_probs_df, time_col, censored_col, threshold=10, title="10-Year Calibration Curve", save_folder=None, show_plot=False, keyword=None):
    """
    Plot calibration curves for models, ensuring consistent and repeatable results.

    Args:
        models_to_plot (list): List of model names to include in the calibration plot.
        all_probs_df (pd.DataFrame): DataFrame containing predicted probabilities and survival data.
        time_col (str): Name of the column representing time-to-event.
        censored_col (str): Name of the column representing censoring status (1 = event, 0 = censored).
        threshold (float): Time threshold (e.g., 10 years).
        title (str): Title for the plot.
        save_folder (str, optional): Folder to save the plot as a .png file.
    """
    plt.figure(figsize=(8, 6))

    # Create Actual Outcome Column
    all_probs_df['Actual Outcome'] = ((all_probs_df[time_col] <= threshold) & (all_probs_df[censored_col] == 1)).astype(int)

    # Loop through models and plot calibration curves
    for model_name in models_to_plot:
        if model_name in all_probs_df.columns:
            # Get predicted probabilities and actual outcomes
            predicted = all_probs_df[model_name].values
            actual = all_probs_df['Actual Outcome'].values

            # Ensure deterministic binning for calibration curve
            sorted_indices = np.argsort(predicted)
            predicted_sorted = predicted[sorted_indices]
            actual_sorted = actual[sorted_indices]

            # Compute calibration curve
            prob_true, prob_pred = calibration_curve(
                y_true=actual_sorted,
                y_prob=predicted_sorted,
                n_bins=10,
                strategy='uniform'
            )

            prob_true = 1 - prob_true

            # Plot the calibration curve
            plt.plot(prob_pred, prob_true, marker='o', label=model_name_map.get(model_name, model_name), color=color_list[models_to_plot.index(model_name)])

    # Customize the plot
    keyword_str = ' '.join(keyword).replace('_', ' ')
    plt.title(f"{title} ({keyword_str})",  fontproperties=font_prop, fontsize=16, pad=10)
    plt.xlabel("Predicted Probability (10 years)", fontsize=14, fontproperties=font_prop)
    plt.ylabel("Observed Proportion (10 years)", fontsize=14, fontproperties=font_prop)

    # Customize legend with a white background
    legend = plt.legend(prop=font_prop, fontsize=13)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot if save_folder is provided
    if save_folder:
        keyword_str = keyword.replace('_', ' ') if keyword else "plot"
        save_path = f"{save_folder}/calibration_curve_{keyword_str}.png"
        plt.savefig(save_path, format='png')

    # Show the plot if show_plot is True
    if show_plot:
        plt.show()
    else:
        plt.close()

def process_folder_calibration(base_dir, keywords, threshold, save_folder, ignore_svm=True):

    # Initialize a DataFrame to store the results
    all_probs_df = pd.DataFrame()

    # Iterate through each folder and compute calibration curves
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

            # Load test data
            test_x = pd.read_pickle(os.path.join(folder_path, 'data', 'test_x.pkl'))
            test_y = pd.read_pickle(os.path.join(folder_path, 'data', 'test_y.pkl'))

            # Generate all probabilities
            time_point = 10  # Time point for analysis
            all_probs_df = generate_all_probabilities(models_list, test_x, test_y['time2event'], test_y['censored'], time_point, cols_x)
    
    # Plot the calibration curve
    models_to_plot = ['cox_ph', 'gboost', 'deepsurv', 'deephit', 'rsf']
    if not ignore_svm:
        models_to_plot.append('svm')
    plot_10_year_calibration_curve(
        models_to_plot=models_to_plot,
        all_probs_df=all_probs_df,
        time_col='time',
        censored_col='outcomeTime',
        threshold=threshold,
        title="10-Year Calibration Plot",
        save_folder=save_folder,
        keyword='_'.join(keywords)
    )

def main():
    parser = argparse.ArgumentParser(description="Compute and plot 10-year calibration curves for models in subfolders.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the base directory.")
    parser.add_argument('--keyword', type=str, required=True, help="Keywords to select folders (e.g., 'SOF_anyfx').")
    parser.add_argument('--threshold', type=int, default=10, help="Time threshold for the calibration plot (default: 10 years).")
    parser.add_argument('--ignore_svm', action='store_true', default=True, help="Ignore the SVM model.")
    args = parser.parse_args()

    base_dir = args.folder
    keywords = args.keyword.split('_')
    threshold = args.threshold
    save_folder = os.path.join(base_dir, 'summary')
    ignore_svm = args.ignore_svm

    process_folder_calibration(base_dir, keywords, threshold, save_folder, ignore_svm)

if __name__ == "__main__":
    main()
