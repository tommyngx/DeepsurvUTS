import os
import pandas as pd
import numpy as np
import argparse
import shap
import torch
import pickle
import matplotlib.pyplot as plt
from utils import loading_config
from evaluate import load_models_and_results


def reverse_scaling(X_scaled, scaler, feature_names):
    X_original = scaler.inverse_transform(X_scaled)
    return pd.DataFrame(X_original, columns=feature_names)

def plot_shap_values_for_ml_model(model, X_train, y_train, X_val, scaler, cols_x, save_folder=None):
    """
    Compute and plot SHAP values for a machine learning model.

    Args:
        model: Trained machine learning model.
        X_train (pd.DataFrame): Scaled training feature data.
        y_train (pd.Series): Training target values.
        X_val (pd.DataFrame): Scaled validation feature data.
        scaler: Fitted scaler used for preprocessing features.
        cols_x (list): List of feature column names.
        save_folder (str, optional): Folder to save the SHAP plots as .png files.

    Returns:
        None
    """
    # Reverse scaling for SHAP interpretation
    X_train_original = reverse_scaling(X_train[cols_x], scaler, feature_names=cols_x)
    X_val_original = reverse_scaling(X_val[cols_x], scaler, feature_names=cols_x)

    # Fit the model
    print(f"Training model: {model.__class__.__name__}...")
    model.fit(X_train_original, y_train)

    # Initialize SHAP Explainer
    print("Initializing SHAP explainer...")
    explainer = shap.Explainer(model.predict, X_train_original)

    # Compute SHAP values for validation dataset
    print("Computing SHAP values for the validation dataset...")
    shap_values_val = explainer(X_val_original)

    # Plot SHAP waterfall plot for the first validation sample
    print("Generating SHAP waterfall plot for the first validation sample...")
    shap.plots.waterfall(shap_values_val[0])
    if save_folder:
        shap.plots.waterfall(shap_values_val[0], show=False)
        #save_path = f"{save_folder}/results/shap_waterfall.png"
        save_path = f"{save_folder}/shap_waterfall.png"
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=200)
        print(f"SHAP waterfall plot saved at: {save_path}")

    # Plot SHAP summary plot for validation dataset
    print("Generating SHAP summary plot for validation dataset...")
    shap.summary_plot(shap_values_val, X_val_original, show=False)
    if save_folder:
        #save_path = f"{save_folder}/results/shap_summary.png"
        save_path = f"{save_folder}/shap_summary.png"
        plt.savefig(save_path, format='png')
        print(f"SHAP summary plot saved at: {save_path}")

    # Plot SHAP dependence plot for the most important feature
    top_feature = X_val_original.columns[np.argmax(shap_values_val.values.mean(axis=0))]
    print(f"Generating SHAP dependence plot for the top feature: {top_feature}")
    shap.dependence_plot(top_feature, shap_values_val.values, X_val_original, show=False)
    if save_folder:
        #save_path = f"{save_folder}/results/shap_dependence.png"
        save_path = f"{save_folder}/shap_dependence.png"
        plt.savefig(save_path, format='png')
        print(f"SHAP dependence plot saved at: {save_path}")
    return explainer, X_val_original, shap_values_val


# Updated function for SHAP with time interpolation
def plot_shap_values_for_deepsurv(model, X_train, X_val, scaler, cols_x, times):
    """
    Compute and plot SHAP values for PyCox DeepSurv models with time interpolation.

    Args:
        model: Trained PyCox DeepSurv model.
        X_train (pd.DataFrame): Scaled training feature data.
        X_val (pd.DataFrame): Scaled validation feature data.
        scaler: Fitted scaler used for preprocessing features.
        cols_x (list): List of feature column names.
        times (list or np.ndarray): Time points to aggregate survival probabilities.

    Returns:
        None
    """
    # Reverse scaling for SHAP interpretability
    print("Reversing scaling for interpretability...")
    X_train_original = reverse_scaling(X_train[cols_x], scaler, feature_names=cols_x)
    X_val_original = reverse_scaling(X_val[cols_x], scaler, feature_names=cols_x)

    # Wrap the PyCox model's prediction method for SHAP
    def model_predict(X):
        """
        Custom prediction function for SHAP.
        Predicts interpolated survival probabilities at specified time points.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)  # Convert to PyTorch tensor
        survival_preds = model.predict_surv_df(X_tensor)  # Predict survival curves

        # Interpolate survival probabilities at the specified times
        interpolated_probs = survival_preds.reindex(
            survival_preds.index.union(times)
        ).interpolate(method="index").loc[times]

        # Average over time points for SHAP
        mean_probs = interpolated_probs.mean(axis=0)
        return mean_probs.values

    # Initialize SHAP KernelExplainer
    print("Initializing SHAP KernelExplainer...")
    explainer = shap.Explainer(model_predict, X_train_original.values)

    # Compute SHAP values for the validation dataset
    print("Computing SHAP values for validation dataset...")
    shap_values_val = explainer(X_val_original.values)

    # Plot SHAP waterfall plot for the first validation sample
    print("Generating SHAP waterfall plot for the first validation sample...")
    shap.waterfall_plot(shap_values_val[0], feature_names=cols_x)
    plt.show()

    # Plot SHAP summary plot for the entire validation dataset
    print("Generating SHAP summary plot for the validation dataset...")
    shap.summary_plot(shap_values_val, features=X_val_original, feature_names=cols_x)

    # Plot SHAP dependence plot for the most important feature
    top_feature = cols_x[np.argmax(abs(shap_values_val.values).mean(axis=0))]
    print(f"Generating SHAP dependence plot for the top feature: {top_feature}")
    shap.dependence_plot(top_feature, shap_values_val.values, X_val_original, feature_names=cols_x)


def process_folder_shap(base_dir, keywords):
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

            # Load models and results
            models_list, results_table = load_models_and_results(folder_path, cols_x)

            # Load train and test data
            train_x = pd.read_pickle(os.path.join(folder_path, 'data', 'train_x.pkl'))
            train_y = pd.read_pickle(os.path.join(folder_path, 'data', 'train_y.pkl'))
            test_x = pd.read_pickle(os.path.join(folder_path, 'data', 'test_x.pkl'))

            # Load scaler
            with open(f'{folder_path}/data/scaler.pkl', 'rb') as file:
                scaler = pickle.load(file)

            # Ensure the summary directory exists
            summary_dir = os.path.join(base_dir, 'summary')
            os.makedirs(summary_dir, exist_ok=True)

            # Generate SHAP plots for the 'cox_ph' model
            explainer_exp, x1, y1 = plot_shap_values_for_ml_model(
                model=models_list['cox_ph'],
                X_train=train_x,
                y_train=train_y,
                X_val=test_x,
                cols_x=cols_x,
                scaler=scaler,
                save_folder=summary_dir
            )

def main():
    parser = argparse.ArgumentParser(description="Generate SHAP plots for models in subfolders.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the base directory.")
    parser.add_argument('--keyword', type=str, required=True, help="Keywords to select folders (e.g., 'SOF_anyfx').")
    args = parser.parse_args()

    base_dir = args.folder
    keywords = args.keyword.split('_')

    process_folder_shap(base_dir, keywords)

if __name__ == "__main__":
    main()
