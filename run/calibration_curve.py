import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import requests
from sklearn.calibration import calibration_curve
from evaluate import load_models_and_results, generate_all_probabilities

def plot_10_year_calibration_curve(models_to_plot, all_probs_df, time_col, censored_col, threshold, title, save_folder):
    """
    Plot 10-year calibration curves for multiple models.

    Args:
        models_to_plot (list): List of model names to plot.
        all_probs_df (pd.DataFrame): DataFrame containing predicted probabilities and actual outcomes.
        time_col (str): Column name for the time-to-event variable.
        censored_col (str): Column name for the censoring variable.
        threshold (int): Time threshold for the calibration plot (e.g., 10 years).
        title (str): Title of the plot.
        save_folder (str): Folder to save the plot as a .png file.
    """
    # Download and set the custom font
    font_url = 'https://github.com/tommyngx/style/blob/main/Poppins.ttf?raw=true'
    font_path = 'Poppins.ttf'
    response = requests.get(font_url)
    with open(font_path, 'wb') as f:
        f.write(response.content)
    font_prop = FontProperties(fname=font_path, size=19)

    # Model name mapping
    model_name_map = {
        'deepsurv': 'DeepSurv', 'deephit': 'DeepHit',
        'cox_ph': 'CoxPH', 'gboost': 'GradientBoosting',
        'svm': 'SVM-Surv', 'rsf': "RSurvivalForest", 'kaplan_meier': 'KaplanMeier',
        'random': 'Random'
    }

    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Define the color list
    color_list = [
        "#2ca02c", "#8c564b", "#9467bd", "#d62728", "#ff7f0e",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    plt.figure(figsize=(8, 6))

    # Create Actual Outcome Column
    all_probs_df['Actual Outcome'] = ((all_probs_df[time_col] <= threshold) & (all_probs_df[censored_col] == 1)).astype(int)

    # Loop through models and plot calibration curves
    for idx, model in enumerate(models_to_plot):
        color = color_list[idx % len(color_list)]  # Assign color from the list
        display_name = model_name_map.get(model, model)  # Use model_name_map if provided, otherwise use original names

        # Filter the DataFrame for the current model
        model_df = all_probs_df[all_probs_df['model'] == model]

        # Get predicted probabilities and actual outcomes
        predicted = model_df['predicted_prob'].values
        actual = model_df['Actual Outcome'].values

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
        plt.plot(prob_pred, prob_true, marker='o', label=display_name, color=color)

    # Customize the plot
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.title(title, fontproperties=font_prop, pad=20)
    plt.xlabel("Predicted Probability", fontsize=14, fontproperties=font_prop)
    plt.ylabel("Observed Probability", fontsize=14, fontproperties=font_prop)

    # Customize legend with a white background
    legend = plt.legend()
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(save_folder, '10_year_calibration_curve.png')
    plt.savefig(save_path, format='png')
    plt.show()

def process_folder_calibration(base_dir, keywords, threshold, save_folder):
    # Define columns for different sets
    cols_22 = ['age', 'education', 'weight', 'height', 'smoke', 'drink', 'no_falls', 'fx50', 'physical',
               'hypertension', 'copd', 'parkinson', 'cancer', 'rheumatoid', 'cvd',
               'renal', 'depression', 'diabetes', 'Tscore', 'protein', 'calcium', 'coffee']
    cols_11 = ['age', 'weight', 'height', 'fx50', 'smoke', 'drink', 'rheumatoid', 'Tscore']
    cols_5 = ['age', 'weight', 'no_falls', 'fx50', 'Tscore']

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
    models_to_plot = all_probs_df['model'].unique()
    plot_10_year_calibration_curve(
        models_to_plot=models_to_plot,
        all_probs_df=all_probs_df,
        time_col='time',
        censored_col='outcomeTime',
        threshold=threshold,
        title="10-Year Calibration Plot",
        save_folder=save_folder
    )

def main():
    parser = argparse.ArgumentParser(description="Compute and plot 10-year calibration curves for models in subfolders.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the base directory.")
    parser.add_argument('--keyword', type=str, required=True, help="Keywords to select folders (e.g., 'SOF_anyfx').")
    parser.add_argument('--threshold', type=int, default=10, help="Time threshold for the calibration plot (default: 10 years).")
    args = parser.parse_args()

    base_dir = args.folder
    keywords = args.keyword.split('_')
    threshold = args.threshold
    save_folder = os.path.join(base_dir, 'summary')

    process_folder_calibration(base_dir, keywords, threshold, save_folder)

if __name__ == "__main__":
    main()
