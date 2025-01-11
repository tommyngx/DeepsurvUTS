import os
import pandas as pd
import numpy as np
import argparse
import shap
import pickle
import matplotlib.pyplot as plt
from utils import loading_config
from tqdm import tqdm

def scale_shap_values_np(array, min_value=0.01):
    """
    Scales SHAP values in a NumPy array such that the smallest absolute value in each column 
    is at least `min_value`.

    Args:
        array (np.ndarray): NumPy array containing SHAP values.
        min_value (float): The minimum absolute value desired for SHAP values after scaling (default is 0.01).

    Returns:
        tuple: A tuple containing the scaled NumPy array and a list of scaling factors for each column.
    """
    scaling_factors = []
    scaled_array = array.copy()

    for col_idx in range(array.shape[1]):
        column = array[:, col_idx]
        min_abs_value = np.min(np.abs(column))
        if min_abs_value > 0:
            scaling_factor = np.ceil(np.log10(min_value / min_abs_value))
        else:
            scaling_factor = 0

        scaling_factors.append(10 ** scaling_factor)
        scaled_array[:, col_idx] *= 10 ** scaling_factor

    return scaled_array, scaling_factors

def plot_shap_values_from_explainer(shap_values_val, X_val, save_folder, model_name, font_prop):
    """
    Plot SHAP values from loaded SHAP values and save the plots.

    Args:
        shap_values_val (shap.Explanation): Loaded SHAP values.
        X_val (pd.DataFrame): Validation feature data.
        save_folder (str): Folder to save the SHAP plots as .png files.
        model_name (str): Name of the model.
        font_prop (FontProperties): Font properties for the plot.
    """
    
    shap_values_val.values, _ =  scale_shap_values_np(shap_values_val.values)
    print("shap_values_val: ", shap_values_val)
    # Create list of plots to generate
    plots_to_generate = [
        ('global bar', lambda: shap.plots.bar(shap_values_val, max_display=10, show=False)),
        ('local bar', lambda: shap.plots.bar(shap_values_val[0], show=False)),
        ('waterfall', lambda: shap.plots.waterfall(shap_values_val[0], show=False)),
        ('summary', lambda: shap.summary_plot(shap_values_val, X_val, show=False))
    ]

    # Set up progress bar
    pbar = tqdm(plots_to_generate, desc=f"Generating SHAP plots for {model_name}")
    
    for plot_name, plot_func in pbar:
        pbar.set_description(f"Generating {plot_name} plot for {model_name}")
        plot_func()
        plt.gcf().set_size_inches(11, 6)
        if save_folder:
            save_path = f"{save_folder}/shap_{plot_name.replace(' ', '_')}_{model_name}.png"
            plt.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
            plt.close()

def process_folder_explainer(base_dir, keywords, model):
    # Load configuration
    config, font_prop, model_name_map, color_list, cols_22, cols_11, cols_5 = loading_config()

    # Iterate through each folder and generate SHAP plots
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

            # Load test data
            test_x = pd.read_pickle(os.path.join(folder_path, 'data', 'test_x.pkl'))

            # Ensure the summary directory exists
            summary_dir = os.path.join(base_dir, 'summary', 'xplainer', '_'.join(keywords))
            os.makedirs(summary_dir, exist_ok=True)

            # Load SHAP values
            shap_values_file = f'shap_values_{model}_{"_".join(keywords)}.pkl'
            shap_values_path = os.path.join(base_dir, 'xplainer', shap_values_file)

            if os.path.exists(shap_values_path):
                with open(shap_values_path, 'rb') as f:
                    shap_values_val = pickle.load(f)
                #print(f"Loaded SHAP values for model: {model}")

                # Generate SHAP plots
                plot_shap_values_from_explainer(shap_values_val, test_x[cols_x], summary_dir, model, font_prop)
            else:
                print(f"SHAP values file not found: {shap_values_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate SHAP plots from explainers in subfolders.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the base directory.")
    parser.add_argument('--keyword', type=str, required=True, help="Keywords to select folders (e.g., 'SOF_anyfx').")
    parser.add_argument('--model', type=str, required=True, choices=['gboost', 'coxph', 'deepsurv'], 
                      help="Model to select for SHAP explainer (gboost, coxph, or deepsurv).")
    args = parser.parse_args()

    base_dir = args.folder
    keywords = args.keyword.split('_')
    model = args.model

    process_folder_explainer(base_dir, keywords, model)

if __name__ == "__main__":
    main()
