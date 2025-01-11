import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import requests
from matplotlib.font_manager import FontProperties

def get_csv_files(base_dir, keywords):
    """
    Retrieve CSV files from subfolders that match the specified keywords.

    Args:
        base_dir (str): Base directory to search.
        keywords (list): List of keywords to match folder names.

    Returns:
        pd.DataFrame: Merged DataFrame from the matched CSV files.
    """
    dataframes = []
    for root, dirs, files in os.walk(base_dir):
        if all(keyword in root for keyword in keywords):
            for file in files:
                if file == 'result.csv':
                    csv_path = os.path.join(root, file)
                    df = pd.read_csv(csv_path)
                    folder_name = os.path.basename(os.path.dirname(root))
                    risk_label = folder_name.split('_')[1] + ' risks'
                    df = df[['model', 'score_test']].rename(columns={'score_test': risk_label})
                    dataframes.append(df)
                    #print(f"Loaded DataFrame from {csv_path}")
    if dataframes:
        merged_df = dataframes[0]
        for df in dataframes[1:]:
            merged_df = pd.merge(merged_df, df, on='model', how='outer')
        # Sort columns, excluding 'model'
        sorted_columns = sorted(merged_df.columns.drop('model'), key=lambda x: int(x.split()[0]))
        merged_df = merged_df[['model'] + sorted_columns]
        return merged_df
    else:
        return pd.DataFrame()



def loading_config():
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

    # Column sets
    cols_22 = config['cols_22']
    cols_11 = config['cols_11']
    cols_5 = config['cols_5']

    return config, font_prop, model_name_map, color_list, cols_22, cols_11, cols_5
