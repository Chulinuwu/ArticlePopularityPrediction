import pandas as pd
import os

def concat_csv_files(file_list, output_file):
    df_list = [pd.read_csv(file) for file in file_list]
    concatenated_df = pd.concat(df_list, ignore_index=True)
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated CSV saved to {output_file}")

if __name__ == "__main__":
    input_csv = [
        # 'input_file.csv',
        # 'input_file.csv',
        # 'input_file.csv',
        # 'input_file.csv',
    ]
    output_csv = 'output_file.csv'
    concat_csv_files(input_csv, output_csv)