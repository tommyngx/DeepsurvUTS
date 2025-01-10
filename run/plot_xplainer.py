import os
import pandas as pd
import numpy as np
import argparse
import shap
import pickle
import matplotlib.pyplot as plt
from utils import loading_config

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
    
    # Plot SHAP global bar plot
    print("Generating SHAP global bar plot...")
    shap.plots.bar(shap_values_val, max_display=10, show=False)
    plt.gcf().set_size_inches(11,6)
    if save_folder:
        save_path = f"{save_folder}/shap_global_bar_{model_name}.png"
        plt.savefig(save_path, format='png')
        plt.close()
        print(f"SHAP global bar plot saved at: {save_path}")

    # Plot SHAP local bar plot for the first validation sample
    print("Generating SHAP local bar plot for the first validation sample...")
    shap.plots.bar(shap_values_val[0],  show=False)
    plt.gcf().set_size_inches(11,6)
    if save_folder:
        save_path = f"{save_folder}/shap_local_bar_{model_name}.png"
        plt.savefig(save_path, format='png')
        plt.close()
        print(f"SHAP local bar plot saved at: {save_path}")

    # Plot SHAP waterfall plot for the first validation sample
    print("Generating SHAP waterfall plot for the first validation sample...")
    shap.plots.waterfall(shap_values_val[0])
    if save_folder:
        shap.plots.waterfall(shap_values_val[0], show=False)
        save_path = f"{save_folder}/shap_waterfall_{model_name}.png"
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=200)
        plt.close()
        print(f"SHAP waterfall plot saved at: {save_path}")

    # Plot SHAP summary plot for validation dataset
    print("Generating SHAP summary plot for validation dataset...")
    shap.summary_plot(shap_values_val, X_val, show=False)
    if save_folder:
        save_path = f"{save_folder}/shap_summary_{model_name}.png"
        plt.savefig(save_path, format='png')
        plt.close()
        print(f"SHAP summary plot saved at: {save_path}")

    # Plot SHAP dependence plot for the most important feature
    #top_feature = X_val.columns[np.argmax(shap_values_val.values.mean(axis=0))]
    #print(f"Generating SHAP dependence plot for the top feature: {top_feature}")
    #shap.dependence_plot(top_feature, shap_values_val.values, X_val, show=False)
    #if save_folder:
    #    save_path = f"{save_folder}/shap_dependence_{model_name}.png"
    #    plt.savefig(save_path, format='png')
    #    plt.close()
    #    print(f"SHAP dependence plot saved at: {save_path}")

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
            summary_dir = os.path.join(base_dir, 'summary')
            os.makedirs(summary_dir, exist_ok=True)

            # Load SHAP values
            shap_values_file = f'shap_values_{model}_{"_".join(keywords)}.pkl'
            shap_values_path = os.path.join(base_dir, 'xplainer', shap_values_file)

            if os.path.exists(shap_values_path):
                with open(shap_values_path, 'rb') as f:
                    shap_values_val = pickle.load(f)
                print(f"Loaded SHAP values for model: {model}")

                # Generate SHAP plots
                plot_shap_values_from_explainer(shap_values_val, test_x[cols_x], summary_dir, model, font_prop)
            else:
                print(f"SHAP values file not found: {shap_values_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate SHAP plots from explainers in subfolders.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the base directory.")
    parser.add_argument('--keyword', type=str, required=True, help="Keywords to select folders (e.g., 'SOF_anyfx').")
    parser.add_argument('--model', type=str, required=True, choices=['gboost', 'coxph'], help="Model to select for SHAP explainer (gboost or coxph).")
    args = parser.parse_args()

    base_dir = args.folder
    keywords = args.keyword.split('_')
    model = args.model

    process_folder_explainer(base_dir, keywords, model)

if __name__ == "__main__":
    main()
