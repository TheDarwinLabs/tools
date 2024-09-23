import pandas as pd
import wbgapi as wb
import os
import requests
from bs4 import BeautifulSoup

# Add this dictionary after the imports
additional_populations = {
    "FJI": 936240,
    "SAH": 0,
    "BHS": 409984,
    "FLK": 3780,
    "GRL": 56877,
    "ATF": 0,
    "TLS": 1360596,
    "BLZ": 419199,
    "GUY": 813834,
    "SUR": 623236,
    "GNQ": 1714671,
    "SWZ": 1210822,
    "PSX": 5371230,
    "VUT": 334506,
    "BTN": 787424,
    "LVA": 1830211,
    "EST": 1322765,
    "LUX": 654768,
    "NCL": 289950,
    "SLB": 740424,
    "TWN": 23888595,
    "ISL": 375318,
    "BRN": 452524,
    "ATA": 0,
    "CYN": 0,
    "CYP": 1260138,
    "DJI": 1120849,
    "SOL": 0,
    "MKD": 2085679,
    "MNE": 626485,
    "KOS": 1761985,
    "TTO": 1406585,
    "SDS": 0
}

def get_population_data():
    cache_file = 'country_populations_cache.csv'
    
    if os.path.exists(cache_file):
        print("Reading population data from cache...")
        return pd.read_csv(cache_file)
    
    try:
        print("Downloading population data...")
        # Fetch population data for all countries, most recent year
        data = wb.data.DataFrame('SP.POP.TOTL', mrv=1)
        
        # Reset index to convert multi-index to columns
        data = data.reset_index()
        
        # Rename columns
        data.columns = ['code', 'population']
        
        # Convert population to integer, replacing NaN with 0
        data['population'] = data['population'].fillna(0).astype(int)
        
        # Keep only the code and population columns
        data = data[['code', 'population']]
        
        # Cache the data
        data.to_csv(cache_file, index=False)
        
        print(f"Successfully fetched population data for {len(data)} countries.")
        return data
    
    except Exception as e:
        print(f"Error downloading or processing data: {e}")
        return pd.DataFrame()

def search_population_online(country_code):
    try:
        url = f"https://www.worldometers.info/world-population/population-by-country/"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'example2'})
        rows = table.find_all('tr')
        
        for row in rows[1:]:  # Skip header row
            columns = row.find_all('td')
            if columns[1].text.strip() == country_code:
                population = int(columns[2].text.strip().replace(',', ''))
                return population
        
        return 0
    except Exception as e:
        print(f"Error searching population online for {country_code}: {e}")
        return 0

def update_zero_populations(df):
    zero_pop_countries = df[df['population'] == 0]
    for _, row in zero_pop_countries.iterrows():
        population = search_population_online(row['code'])
        if population > 0:
            df.loc[df['code'] == row['code'], 'population'] = population
            print(f"Updated population for {row['code']}: {population}")
    
    # Add this new loop to update with additional populations
    for code, population in additional_populations.items():
        if code not in df['code'].values:
            df = pd.concat([df, pd.DataFrame({'code': [code], 'population': [population]})], ignore_index=True)
            print(f"Added population for {code}: {population}")
        elif df.loc[df['code'] == code, 'population'].iloc[0] == 0:
            df.loc[df['code'] == code, 'population'] = population
            print(f"Updated population for {code}: {population}")
    return df

def create_population_csv():
    df = get_population_data()
    if not df.empty:
        df = update_zero_populations(df)
        df.to_csv('country_populations.csv', index=False)
        print("country_populations.csv has been created successfully.")
        print(f"Number of countries in population data: {len(df)}")
    else:
        print("Failed to create country_populations.csv due to data retrieval issues.")

if __name__ == "__main__":
    create_population_csv()
