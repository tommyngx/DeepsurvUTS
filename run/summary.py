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
                    risk_label = folder_name.split('_')[1] + ' risks'
                    df = df[['model', 'score_test']].rename(columns={'score_test': risk_label})
                    dataframes.append(df)
                    print(f"Loaded DataFrame from {csv_path}")
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

def plot_performance_benchmark(df, summary_dir):
    """
    Plot a scatter plot of the performance benchmark from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the performance data.
        summary_dir (str): Directory to save the summary image.
    """
    df.set_index('model', inplace=True)
    
    fig, ax = plt.subplots()
    for model in df.index:
        x = df.loc[model]
        y = df.loc[model]
        ax.plot(x, y, marker='o', label=model)  # Draw lines connecting dots
        for i, txt in enumerate(df.columns):
            ax.annotate(f"{txt} ({y.iloc[i]:.2f})", (x.iloc[i], y.iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title('Performance Benchmark')
    plt.xlabel('Score')
    plt.ylabel('Score')
    plt.legend(title='Model')
    plt.grid(True)
    
    # Create summary directory if it does not exist
    os.makedirs(summary_dir, exist_ok=True)
    image_path = os.path.join(summary_dir, 'performance_benchmark.png')
    plt.savefig(image_path)  # Save the plot as an image
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Retrieve and print CSV files from subfolders.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the base directory.")
    parser.add_argument('--keyword', type=str, required=True, help="Keywords to select folders (e.g., 'SOF_anyfx').")
    args = parser.parse_args()

    base_dir = args.folder
    keywords = args.keyword.split('_')
    summary_dir = os.path.join(base_dir, 'summary')

    merged_df = get_csv_files(base_dir, keywords)
    if not merged_df.empty:
        print(merged_df)
        plot_performance_benchmark(merged_df, summary_dir)
    else:
        print("No matching CSV files found.")

if __name__ == "__main__":
    main()
