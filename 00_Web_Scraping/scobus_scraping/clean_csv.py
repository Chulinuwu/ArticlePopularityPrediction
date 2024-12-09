import pandas as pd

def clean_csv(path) :
    df = pd.read_csv(path)
    df['Author Keywords'] = df['Author Keywords'].str.replace(';', ',', regex=False)
    df[['Title','Source title','Cited by','Authors Country','Author Keywords', 'Year']].to_csv('output_file_filtered.csv', index=False) # Change 'output_file_filtered.csv'

if __name__ == "__main__":
    path = 'input_file.csv' # Change 'input_file.csv'
    clean_csv(path)