import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import requests
import yaml
from sklearn.metrics import roc_curve, auc
from evaluate import load_models_and_results, generate_all_probabilities

# Load configuration from YAML file
with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

# Set font properties
font_url = config['font']['url']
font_path = config['font']['path']
response = requests.get(font_url)
with open(font_path, 'wb') as f:
    f.write(response.content)
font_prop = FontProperties(fname=font_path, size=config['font']['size'])

# Model name mapping
model_name_map = config['model_name_map']

# Color list
color_list = config['color_list']

def plot_roc_curve(models_to_plot, all_probs_df, time_col, censored_col, threshold=10, title="10-Year ROC Curve", save_folder=None):
    """
    Plot ROC curves for models with support for mapped names, ensuring colors follow a specific order.

    Args:
        models_to_plot (list): List of model names to include in the ROC plot.
        all_probs_df (pd.DataFrame): DataFrame containing predicted probabilities and survival data.
        time_col (str): Name of the column representing time-to-event.
        censored_col (str): Name of the column representing censoring status (1 = event, 0 = censored).
        threshold (float): Time threshold (e.g., 10 years).
        title (str): Title for the plot.
        save_folder (str, optional): Folder to save the plot as a .png file.
    """
    # Step 1: Create Actual Outcome Column
    all_probs_df['Actual Outcome'] = ((all_probs_df[time_col] <= threshold) & (all_probs_df[censored_col] == 1)).astype(int)

    plt.figure(figsize=(8, 6))

    # Step 2: Loop through models and plot ROC curves
    for model_name in models_to_plot:
        if model_name in all_probs_df.columns:
            # Map model name if mapping is provided
            display_name = model_name_map.get(model_name, model_name)

            # Get predicted probabilities and actual outcomes
            predicted = all_probs_df[model_name]
            actual = all_probs_df['Actual Outcome']

            actual = 1 - actual
            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(y_true=actual, y_score=predicted)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.plot(
                fpr, tpr,
                label=f"{display_name} (AUC = {roc_auc:.2f})",
                color=color_list[models_to_plot.index(model_name)]
            )

    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', alpha=0.7)

    # Customize plot
    plt.title(title, fontproperties=font_prop, fontsize=16, pad=10)
    plt.xlabel("1 - Specificity", fontsize=14, fontproperties=font_prop)
    plt.ylabel("Sensitivity", fontsize=14, fontproperties=font_prop)
    plt.legend(loc="best", prop=font_prop, fontsize=13)

    # Customize legend with a white background
    legend = plt.legend(prop=font_prop, fontsize=13)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    # Add markers with lines to the legend
    #for handle in legend.legendHandles:
    #    handle.set_marker('o')
    #    handle.set_linestyle('-')

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot if save_folder is provided
    if save_folder:
        save_path = f"{save_folder}/roc_curve.png"
        plt.savefig(save_path, format='png')

    plt.show()

def process_folder_roc(base_dir, keywords, threshold, save_folder, ignore_svm=True):
    # Define columns for different sets
    cols_22 = ['age', 'education', 'weight', 'height', 'smoke', 'drink', 'no_falls', 'fx50', 'physical',
               'hypertension', 'copd', 'parkinson', 'cancer', 'rheumatoid', 'cvd',
               'renal', 'depression', 'diabetes', 'Tscore', 'protein', 'calcium', 'coffee']
    cols_11 = ['age', 'weight', 'height', 'fx50', 'smoke', 'drink', 'rheumatoid', 'Tscore']
    cols_5 = ['age', 'weight', 'no_falls', 'fx50', 'Tscore']

    # Initialize a DataFrame to store the results
    all_probs_df = pd.DataFrame()

    # Iterate through each folder and compute ROC curves
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
    
    # Plot the ROC curve
    models_to_plot = ['cox_ph', 'gboost', 'deepsurv', 'deephit', 'rsf']
    if not ignore_svm:
        models_to_plot.append('svm')
    plot_roc_curve(
        models_to_plot=models_to_plot,
        all_probs_df=all_probs_df,
        time_col='time',
        censored_col='outcomeTime',
        threshold=threshold,
        title="10-Year ROC Curve",
        save_folder=save_folder
    )

def main():
    parser = argparse.ArgumentParser(description="Compute and plot 10-year ROC curves for models in subfolders.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the base directory.")
    parser.add_argument('--keyword', type=str, required=True, help="Keywords to select folders (e.g., 'SOF_anyfx').")
    parser.add_argument('--threshold', type=int, default=10, help="Time threshold for the ROC plot (default: 10 years).")
    parser.add_argument('--ignore_svm', action='store_true', default=True, help="Ignore the SVM model.")
    args = parser.parse_args()

    base_dir = args.folder
    keywords = args.keyword.split('_')
    threshold = args.threshold
    save_folder = os.path.join(base_dir, 'summary')
    ignore_svm = args.ignore_svm

    process_folder_roc(base_dir, keywords, threshold, save_folder, ignore_svm)

if __name__ == "__main__":
    main()
