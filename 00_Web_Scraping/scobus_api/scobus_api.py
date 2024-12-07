import requests
import json
import time

# Replace this with your Scopus API Key
SCOPUS_API_KEY = "a900b8e4b711f9b9ace8db00011182b0"

# Scopus API base URL
BASE_URL = "https://api.elsevier.com/content/search/scopus"

# Parameters for Scopus API
START_YEAR = 2018  # Example: Start year of the range
END_YEAR = 2023    # Example: End year of the range
NUM_PAPERS = 25  # Target number of papers
RESULTS_PER_REQUEST = 25  # Maximum results per request (as per Scopus API)

def fetch_scopus_data_by_year(api_key, start_year, end_year, count):
    """
    Fetch data from Scopus API by year range and return a list of research paper metadata.
    """
    headers = {
        "X-ELS-APIKey": api_key
    }

    papers = []
    start = 0  # Start index for pagination

    while len(papers) < count:
        # Adjust the count for the final request
        fetch_count = min(RESULTS_PER_REQUEST, count - len(papers))

        # Construct the query for year range
        query = f"PUBYEAR > {start_year - 1} AND PUBYEAR < {end_year + 1}"  # Include all years in range

        params = {
            "query": query,
            "start": start,
            "count": fetch_count,
            "field": "dc:title,dc:identifier,prism:doi,authkeywords,indexkeywords,affiliation,prism:publicationName,citedby-count,prism:coverDate",
        }

        try:
            # Make API request
            response = requests.get(BASE_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract paper entries
            entries = data.get("search-results", {}).get("entry", [])
            papers.extend(entries)

            # Update start for next pagination
            start += len(entries)

            # Delay to respect API rate limits
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break

    return papers

def save_to_json(data, filename):
    """
    Save a list of data to a JSON file.
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving data: {e}")

# Fetch and save Scopus data by year
if __name__ == "__main__":
    print(f"Fetching data from Scopus API for years {START_YEAR} to {END_YEAR}...")
    research_papers = fetch_scopus_data_by_year(SCOPUS_API_KEY, START_YEAR, END_YEAR, NUM_PAPERS)

    # Check if data was fetched successfully
    if research_papers:
        print(f"Fetched {len(research_papers)} papers successfully.")
        save_to_json(research_papers, f"scopus_papers_{START_YEAR}_to_{END_YEAR}.json")
    else:
        print("Failed to fetch research papers.")
