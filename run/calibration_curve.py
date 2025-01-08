import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import requests
from evaluate import load_models_and_results

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

    for idx, model in enumerate(models_to_plot):
        color = color_list[idx % len(color_list)]  # Assign color from the list
        display_name = model_name_map.get(model, model)  # Use model_name_map if provided, otherwise use original names

        # Filter the DataFrame for the current model
        model_df = all_probs_df[all_probs_df['model'] == model]

        # Compute observed and predicted probabilities
        observed = model_df[model_df[time_col] <= threshold][censored_col].mean()
        predicted = model_df[model_df[time_col] <= threshold]['predicted_prob'].mean()

        # Plot the calibration curve
        plt.plot(predicted, observed, marker='o', label=display_name, color=color)

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

            # Compute predicted probabilities for each model
            for model_name, model in models_list.items():
                if hasattr(model, 'predict_surv_df'):
                    surv_df = model.predict_surv_df(test_x[cols_x].values)
                    predicted_prob = 1 - surv_df.loc[threshold].values
                else:
                    predicted_prob = model.predict_proba(test_x[cols_x].values)[:, 1]

                # Append results to the DataFrame
                temp_df = pd.DataFrame({
                    'model': model_name,
                    'time': test_y['time2event'],
                    'outcomeTime': test_y['censored'],
                    'predicted_prob': predicted_prob
                })
                all_probs_df = pd.concat([all_probs_df, temp_df], ignore_index=True)

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
