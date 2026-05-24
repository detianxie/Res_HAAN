import os
import pandas as pd
import numpy as np
import json
import ast

def preprocess_dataset():
    """
    Data pre-processing script：
    1. Read the raw Mendeley dataset CSV
    2. Extract the regression targets, calculate the global mean and standard deviation, and save them as JSON
    3. Standardise the response (Z-score)
    4. Map the classification labels to integers (Good: 0, Bad: 1, Expulsion: 2)
    5. The ”final_data.csv“ file required for training the generative model
    """
    print("=== Start data pre-processing ===")
    
    # Path configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    raw_csv_path = os.path.join(data_dir, "Data_RSW.csv")
    final_csv_path = os.path.join(data_dir, "final_data.csv")
    stats_json_path = os.path.join(data_dir, "regression_stats.json")
    
    # 1. Check whether the raw data exists
    if not os.path.exists(raw_csv_path):
        print(f"[Error] The source data file cannot be found: {raw_csv_path}")
        print("Please first download the dataset from Mendeley Data, save it as Data_RSW.csv, and place it in the data/ directory.")
        return
        
    df_raw = pd.read_csv(raw_csv_path)
    print(f"The raw data has been successfully loaded; there are a total of  {len(df_raw)} records.")
    
    # 2. Calculate the mean and standard deviation of the regression target
    # Assuming that the original regression columns in the Mendeley dataset are named “NuggetDiameter” and 'TensileShearLoad'
    # Please modify the following two variables according to the actual column names in your CSV file
    col_nugget = 'NuggetDiameter (mm)' 
    col_pull = 'PullTest (N)'
    
    stats = {
        'NuggetDiameter (mm)': {
            'mean': float(df_raw[col_nugget].mean()),
            'std': float(df_raw[col_nugget].std())
        },
        'PullTest (N)': {
            'mean': float(df_raw[col_pull].mean()),
            'std': float(df_raw[col_pull].std())
        }
    }
    
    # Save the statistics to a JSON file for later use in training and inference
    with open(stats_json_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Regression task statistics have been saved to: {stats_json_path}")
    
    # 3. Data transformation and mapping
    df_final = pd.DataFrame()
    
    # Retain image path columns (assuming the original column names are these)
    df_final['ir_filename'] = df_raw['ir_filename']
    df_final['rgb_f_filename'] = df_raw['rgb_f_filename']
    df_final['rgb_b_filename'] = df_raw['rgb_b_filename']
    
    # Map labels
    label_map = {'Good': 0, 'Bad': 1, 'Explode': 2}
    df_final['class_label_id'] = df_raw['QualityLabel'].map(label_map)
    
    # Construct regression_label_raw and regression_label_normalized
    raw_regs = []
    norm_regs = []
    for idx, row in df_raw.iterrows():
        r_nugget = row[col_nugget]
        r_pull = row[col_pull]
        
        # List of original values
        raw_regs.append([r_nugget, r_pull])
        
        # List of normalized values (Z-score)
        norm_nugget = (r_nugget - stats['NuggetDiameter (mm)']['mean']) / (stats['NuggetDiameter (mm)']['std'] + 1e-6)
        norm_pull = (r_pull - stats['PullTest (N)']['mean']) / (stats['PullTest (N)']['std'] + 1e-6)
        norm_regs.append([norm_nugget, norm_pull])
        
    # Save as a list in string format for use by "ast.literal_eval" in "train.py"
    df_final['regression_label_raw'] = [str(x) for x in raw_regs]
    df_final['regression_label_normalized'] = [str(x) for x in norm_regs]
    
    # 4. Copy temporal feature columns (assuming the process parameter column names are as follows)
    param_columns = ['Pressure', 'WeldTime', 'Angle', 'Force', 'Current', 'ThickA', 'ThickB']
    for col in param_columns:
        if col in df_raw.columns:
            # Ensure the saved values are also in string format
            df_final[col] = df_raw[col].apply(lambda x: str(x) if isinstance(x, list) else x)
        else:
            print(f"[Warning] Process parameter column not found in the raw data: {col}")
            
    # 5. Save the final data
    df_final.to_csv(final_csv_path, index=False)
    print(f"Preprocessing completed! The final training metadata has been saved to: {final_csv_path}")
    print("=== You can now run python train.py to start training ===")

if __name__ == "__main__":
    preprocess_dataset()