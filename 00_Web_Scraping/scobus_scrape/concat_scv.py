import pandas as pd
import os

def concat_csv_files(file_list, output_file):
    # Read and concatenate CSV files
    df_list = [pd.read_csv(file) for file in file_list]
    concatenated_df = pd.concat(df_list, ignore_index=True)
    
    # Save the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated CSV saved to {output_file}")

if __name__ == "__main__":
    
    # List of CSV files to concatenate
    csv_files = [
        './e2500.csv',
        './i2500.csv',
        './o2500.csv',
        './u2500.csv',
    ]
    
    # Output file path
    output_csv = './10000.csv'
    
    concat_csv_files(csv_files, output_csv)