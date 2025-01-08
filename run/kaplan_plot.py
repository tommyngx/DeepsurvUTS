import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from matplotlib.ticker import PercentFormatter
import argparse
from utils import loading_config
from evaluate import load_models_and_results
from sksurv.functions import StepFunction

def plot_kaplan_meier_with_models(y_time, y_censored, models, X_test, models_to_plot, cols_x, time_points, model_name_map=None, title='Kaplan-Meier vs Models Survival Curve', save_folder=None):
    """
    Plot Kaplan-Meier survival curve alongside survival curves from selected models with specified colors.

    Args:
        y_time (pd.Series): Time-to-event data.
        y_censored (pd.Series): Censoring indicator (1 = event, 0 = censored).
        models (dict): Dictionary of trained survival models.
        X_test (pd.DataFrame): Feature data for the test set.
        models_to_plot (list): List of model names to include in the plot.
        cols_x (list): List of feature column names used during model training.
        time_points (np.array): Time points for evaluating survival probabilities.
        model_name_map (dict, optional): Mapping of model names to display-friendly names.
        title (str): Title for the plot.
        save_folder (str, optional): Folder to save the plot as a .png file.
    """
    # Kaplan-Meier estimator
    kmf = KaplanMeierFitter()
    kmf.fit(durations=y_time, event_observed=y_censored)

    # Generate survival probabilities for selected models
    survs = {}
    for model_name in models_to_plot:
        if model_name not in models:
            print(f"Model '{model_name}' not found. Skipping.")
            continue
        model = models[model_name]
        if model_name in ['deepsurv', 'deephit']:
            survs[model_name] = model.predict_surv_df(np.array(X_test[cols_x]).astype(np.float32))
        elif hasattr(model, 'predict_survival_function'):
            survs[model_name] = model.predict_survival_function(X_test[cols_x])

        # Convert SVM risk scores into StepFunction survival probabilities
        elif model_name == 'svm':
            risks = model.predict(X_test[cols_x])  # Predict risk scores
            risks = np.asarray(risks)  # Ensure risks are in array format

            # Normalize risks to [0, 1]
            survival_probs = 1 - np.clip((risks - risks.min()) / (risks.max() - risks.min()), 0, 1)

            # Convert normalized survival probabilities to StepFunctions
            survs[model_name] = [
                StepFunction(time_points, np.full(len(time_points), prob)) for prob in survival_probs
            ]
        else:
            print(f"Model '{model_name}' does not support survival probabilities. Skipping.")

    # Plot Kaplan-Meier survival curve
    plt.figure(figsize=(8, 5))
    kmf.plot_survival_function(label='Kaplan-Meier', ci_show=False, color='black', linestyle='--')

    # Define a list of colors for models
    color_list = [
        "#f0614a", "#11cd9a", "#ab63fa", "#8c564b", '#1f77b4', '#e377c2',
        "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # Create a color map for models
    color_map = {model_name: color_list[idx % len(color_list)] for idx, model_name in enumerate(models_to_plot)}

    # Adjust time_points based on maximum time range in survival curves
    max_time = 25
    for model_name, model_surv in survs.items():
        if isinstance(model_surv, np.ndarray) and isinstance(model_surv[0], StepFunction):
            max_time = min(max_time, min([fn.domain[1] for fn in model_surv]))
        elif isinstance(model_surv, pd.DataFrame):
            max_time = min(max_time, model_surv.index.max())
    time_points = np.linspace(0, max_time, len(time_points))

    # Plot survival curves for models
    for model_name, model_surv in survs.items():
        display_name = model_name_map.get(model_name, model_name) if model_name_map else model_name
        color = color_map[model_name]
        
        if isinstance(model_surv, np.ndarray) and isinstance(model_surv[0], StepFunction):
            # StepFunction survival curves
            evaluated_survs = np.array([fn(time_points) for fn in model_surv])
            mean_surv = evaluated_survs.mean(axis=0)
            plt.plot(
                time_points, mean_surv,
                label=f"{display_name} (Mean Curve)",
                linestyle='-', alpha=0.8, color=color
            )
        elif isinstance(model_surv, pd.DataFrame):
            # DataFrame survival curves
            mean_surv = model_surv.mean(axis=1)
            plt.plot(
                model_surv.index, mean_surv,
                label=f"{display_name} (Mean Curve)",
                linestyle='-', alpha=0.8, color=color
            )
        else:
            print(f"Unsupported format for model '{model_name}'.")

    # Customize the plot
    plt.xlim(0, max_time)
    plt.ylim(0, 1)
    plt.title(title, fontsize=14)
    plt.xlabel('Time (years)', fontsize=12)
    plt.ylabel('Survival Probability', fontsize=12)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))  # Percentage format for survival probability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Customize legend with a white background
    legend = plt.legend(loc='best', fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    plt.tight_layout()

    # Save the plot if save_folder is provided
    if save_folder:
        save_path = f"{save_folder}/results/kaplan_meier.png"
        plt.savefig(save_path, format='png')

    plt.show()

def process_folder_kaplan(base_dir, keywords):
    # Load configuration
    config, font_prop, model_name_map, color_list, cols_22, cols_11, cols_5 = loading_config()

    # Iterate through each folder and generate Kaplan-Meier plots
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and all(keyword in folder for keyword in keywords):
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

            # Generate Kaplan-Meier plot
            time_points = np.linspace(0, 23, 230)
            models_to_plot = ['cox_ph', 'gboost', 'deepsurv', 'deephit', 'rsf']
            plot_kaplan_meier_with_models(
                y_time=test_y['time2event'], y_censored=test_y['censored'],
                models=models_list, X_test=test_x, models_to_plot=models_to_plot,
                cols_x=cols_x, time_points=time_points, model_name_map=model_name_map,
                title=f"Kaplan-Meier vs Models Survival Curve ({folder})", save_folder=folder_path
            )

def main():
    parser = argparse.ArgumentParser(description="Generate Kaplan-Meier plots for models in subfolders.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the base directory.")
    parser.add_argument('--keyword', type=str, required=True, help="Keywords to select folders (e.g., 'SOF_anyfx').")
    args = parser.parse_args()

    base_dir = args.folder
    keywords = args.keyword.split('_')

    process_folder_kaplan(base_dir, keywords)

if __name__ == "__main__":
    main()
