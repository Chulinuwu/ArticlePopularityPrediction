import pandas as pd
import pycountry
import pandas as pd

def extract_countries_pycountry(affiliations):
    if pd.isna(affiliations):
        return None
    countries = [country.name for country in pycountry.countries]
    found_countries = [country for country in countries if country in affiliations]
    return ', '.join(found_countries) if found_countries else None

def clean_csv(path) :
    df = pd.read_csv(path)
    df['Author Keywords'] = df['Author Keywords'].str.replace(';', ',', regex=False)
    df['Authors Country'] = df['Authors with affiliations'].apply(extract_countries_pycountry)
    df[['Title','Source title','Cited by','Authors Country','Author Keywords', 'Year']].to_csv('output_file_filtered.csv', index=False) # Change 'output_file_filtered.csv'

if __name__ == "__main__":
    path = 'output_file.csv'
    clean_csv(path)