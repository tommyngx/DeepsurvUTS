import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt

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
                    df = df[['model', 'score_test']].rename(columns={'score_test': f'{folder_name}'})
                    dataframes.append(df)
                    print(f"Loaded DataFrame from {csv_path}")
    if dataframes:
        merged_df = dataframes[0]
        for df in dataframes[1:]:
            merged_df = pd.merge(merged_df, df, on='model', how='outer')
        return merged_df
    else:
        return pd.DataFrame()

def plot_performance_benchmark(df):
    """
    Plot a line plot of the performance benchmark from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the performance data.
    """
    %matplotlib inline
    df.set_index('model', inplace=True)
    df.T.plot(kind='line', marker='o')
    plt.title('Performance Benchmark')
    plt.xlabel('Folders')
    plt.ylabel('Score')
    plt.legend(title='Model')
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Retrieve and print CSV files from subfolders.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the base directory.")
    parser.add_argument('--keyword', type=str, required=True, help="Keywords to select folders (e.g., 'SOF_anyfx').")
    args = parser.parse_args()

    base_dir = args.folder
    keywords = args.keyword.split('_')

    merged_df = get_csv_files(base_dir, keywords)
    if not merged_df.empty:
        print(merged_df)
        plot_performance_benchmark(merged_df)
    else:
        print("No matching CSV files found.")

if __name__ == "__main__":
    main()
