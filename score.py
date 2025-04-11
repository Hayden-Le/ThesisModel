import os
import pandas as pd
import matplotlib.pyplot as plt

def load_custom_csv(filename, rounds_per_method=13):
    """
    Reads the CSV file with a non-standard structure:
      - One line with the method name (no commas).
      - Next 'rounds_per_method' lines: comma-separated data:
         Version, RMSE, MAE, R2, Timestamp
    Returns a DataFrame with columns: [Method, Version, RMSE, MAE, R2, Timestamp].
    """
    rows = []
    current_method = None
    rounds_count = 0
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                # Skip empty lines
                continue
            
            if ',' not in line:
                # This is a method name line
                current_method = line
                rounds_count = 0
            else:
                # This should be a data line
                parts = line.split(',')
                if len(parts) != 5:
                    print(f"Skipping line (unexpected format): {line}")
                    continue
                version, rmse, mae, r2, timestamp = parts
                try:
                    version = int(version.strip())
                    rmse = float(rmse.strip())
                    mae = float(mae.strip())
                    r2 = float(r2.strip())
                    timestamp = timestamp.strip()
                except Exception as e:
                    print(f"Error processing line '{line}': {e}")
                    continue
                
                rows.append({
                    'Method': current_method,
                    'Version': version,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'Timestamp': timestamp
                })
                
                rounds_count += 1
                # Once we hit 'rounds_per_method' lines for this method,
                # the next method name is expected
                if rounds_count == rounds_per_method:
                    current_method = None
                    rounds_count = 0
    
    df = pd.DataFrame(rows)
    return df

def approach1_r2_offset_normalize(df):
    """
    Applies Approach 1 for R^2:
      1) For each method, find the minimum R^2,
      2) Shift R^2 by subtracting that minimum,
      3) Normalize R^2_sh over [0, 1],
      4) Convert it into a 'loss-like' measure: R2Loss = 1 - NormR2.
    Also normalizes RMSE and MAE in a min–max manner:
      NormRMSE = (RMSE - min(RMSE)) / (max(RMSE) - min(RMSE)), and similarly for MAE.
    Returns a new DataFrame with additional columns:
      NormRMSE, NormMAE, NormR2, R2Loss
    """
    # We'll do the normalization method-by-method, then reassemble.
    all_dfs = []
    for method in df['Method'].unique():
        df_m = df[df['Method'] == method].copy()
        
        # ----- Normalize RMSE -----
        min_rmse = df_m['RMSE'].min()
        max_rmse = df_m['RMSE'].max()
        df_m['NormRMSE'] = (df_m['RMSE'] - min_rmse) / (max_rmse - min_rmse) if max_rmse != min_rmse else 0
        
        # ----- Normalize MAE -----
        min_mae = df_m['MAE'].min()
        max_mae = df_m['MAE'].max()
        df_m['NormMAE'] = (df_m['MAE'] - min_mae) / (max_mae - min_mae) if max_mae != min_mae else 0
        
        # ----- Approach 1 for R^2 -----
        # 1) shift so that min(R^2) becomes 0
        min_r2 = df_m['R2'].min()
        df_m['R2_shifted'] = df_m['R2'] - min_r2
        
        # 2) find max of the shifted R^2
        max_r2_sh = df_m['R2_shifted'].max()
        
        # 3) normalize
        if max_r2_sh != 0:
            df_m['NormR2'] = df_m['R2_shifted'] / max_r2_sh
        else:
            # If everything is the same, set them all to 0
            df_m['NormR2'] = 0
        
        # 4) R2Loss = 1 - NormR2 (lower is better)
        df_m['R2Loss'] = 1 - df_m['NormR2']
        
        all_dfs.append(df_m)
    
    return pd.concat(all_dfs, axis=0).sort_values(by=['Method','Version'])

def compute_composite(df, w_rmse=1/3, w_mae=1/3, w_r2=1/3):
    """
    Computes a composite score for each row in df, using:
      Composite = w_rmse * NormRMSE + w_mae * NormMAE + w_r2 * R2Loss.
    Appends a 'Composite' column to the df and returns it.
    """
    df['Composite'] = (w_rmse * df['NormRMSE']
                       + w_mae * df['NormMAE']
                       + w_r2 * df['R2Loss'])
    return df

def save_plots(df, output_dir="plots"):
    """
    Saves line plots of NormRMSE, NormMAE, R2Loss, and Composite over Version,
    for each method. 
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    metrics = ['NormRMSE', 'NormMAE', 'R2Loss', 'Composite']
    methods = df['Method'].unique()
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for method in methods:
            df_method = df[df['Method'] == method].sort_values(by='Version')
            plt.plot(df_method['Version'], df_method[metric], marker='o', label=method)
        
        plt.xlabel('Global Model Version')
        plt.ylabel(metric)
        plt.title(f'{metric} vs. Global Model Version')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, f"{metric}_vs_Version.png")
        plt.savefig(plot_file)
        print(f"Saved plot to {plot_file}")
        plt.close()

def main():
    # 1) Load the custom CSV
    csv_file = 'final_metric.csv'  # replace if needed
    df_raw = load_custom_csv(csv_file, rounds_per_method=13)
    print("Initial data:\n", df_raw.head())
    
    # 2) Apply Approach 1 offset + normalization for R^2, and min–max for RMSE & MAE
    df_norm = approach1_r2_offset_normalize(df_raw)
    
    # 3) Compute composite (equal weights by default)
    df_final = compute_composite(df_norm, w_rmse=1/3, w_mae=1/3, w_r2=1/3)
    
    # 4) Save final data to a new CSV for reference
    df_final_outfile = "final_metric_normalized.csv"
    df_final.to_csv(df_final_outfile, index=False)
    print(f"Exported normalized data with composite to: {df_final_outfile}")
    
    # 5) Save plots
    save_plots(df_final, output_dir="plots_approach1")

if __name__ == '__main__':
    main()
