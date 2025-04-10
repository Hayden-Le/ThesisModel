import pandas as pd
import matplotlib.pyplot as plt

def load_custom_csv(filename):
    """
    Reads the CSV file that has a non-standard structure:
    - A line containing the method name (no commas) indicates a new block.
    - The next 13 lines contain comma-separated data:
        Version, RMSE, MAE, R2, Timestamp
    Returns a DataFrame with columns:
      'Method', 'Version', 'RMSE', 'MAE', 'R2', 'Timestamp'
    """
    rows = []
    current_method = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Check if the line has commas (data) or not (method name)
            if ',' not in line:
                # New method block
                current_method = line
            elif line:
                # This is a data line; split it by comma
                # Expecting 5 fields: Version, RMSE, MAE, R2, Timestamp
                parts = line.split(',')
                if len(parts) != 5:
                    print("Skipping line (unexpected format):", line)
                    continue
                version, rmse, mae, r2, timestamp = parts
                try:
                    version = int(version.strip())
                    rmse = float(rmse.strip())
                    mae = float(mae.strip())
                    r2 = float(r2.strip())
                    timestamp = timestamp.strip()
                except Exception as e:
                    print("Error processing line:", line, "Error:", e)
                    continue
                # Append the row with the current method
                rows.append({
                    'Method': current_method,
                    'Version': version,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'Timestamp': timestamp
                })
    df = pd.DataFrame(rows)
    return df

def visualize_metrics(df):
    # Print column names to check structure
    print("Columns in DataFrame:", df.columns.tolist())
    
    metrics = ['RMSE', 'MAE', 'R2']
    methods = df['Method'].unique()
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for method in methods:
            df_method = df[df['Method'] == method].sort_values(by='Version')
            plt.plot(df_method['Version'], df_method[metric], marker='o', label=method)
        plt.xlabel('Global Model Version (Round)')
        plt.ylabel(metric)
        plt.title(f'{metric} vs Global Model Version for Different Fed Methods')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Replace 'final_metric.csv' with your CSV file name
    df = load_custom_csv('final_metric.csv')
    print(df.head())  # optional: to check if data was loaded correctly
    visualize_metrics(df)
