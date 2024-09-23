import pandas as pd
import wbgapi as wb
import os

# Add this dictionary after the imports
additional_economies = {
    "FJI": {"GDP": 4.454e9, "GDP_per_capita": 4754.74},
    "SAH": {"GDP": 0, "GDP_per_capita": 0},
    "BHS": {"GDP": 1.291e10, "GDP_per_capita": 31474.15},
    "FLK": {"GDP": 2.97e8, "GDP_per_capita": 78571.43},
    "GRL": {"GDP": 3.08e9, "GDP_per_capita": 54151.89},
    "ATF": {"GDP": 0, "GDP_per_capita": 0},
    "TLS": {"GDP": 2.017e9, "GDP_per_capita": 1482.41},
    "BLZ": {"GDP": 1.987e9, "GDP_per_capita": 4738.69},
    "GUY": {"GDP": 1.5914e10, "GDP_per_capita": 19555.86},
    "SUR": {"GDP": 3.99e9, "GDP_per_capita": 6403.84},
    "GNQ": {"GDP": 1.1027e10, "GDP_per_capita": 6430.76},
    "SWZ": {"GDP": 4.738e9, "GDP_per_capita": 3913.03},
    "PSX": {"GDP": 1.8037e10, "GDP_per_capita": 3357.91},
    "VUT": {"GDP": 9.46e8, "GDP_per_capita": 2827.47},
    "BTN": {"GDP": 2.539e9, "GDP_per_capita": 3224.43},
    "LVA": {"GDP": 4.1391e10, "GDP_per_capita": 22610.75},
    "EST": {"GDP": 3.8100e10, "GDP_per_capita": 28799.07},
    "LUX": {"GDP": 8.4077e10, "GDP_per_capita": 128377.63},
    "NCL": {"GDP": 9.45e9, "GDP_per_capita": 32592.17},
    "SLB": {"GDP": 1.591e9, "GDP_per_capita": 2149.95},
    "TWN": {"GDP": 7.8945e11, "GDP_per_capita": 33046.37},
    "ISL": {"GDP": 2.7835e10, "GDP_per_capita": 74142.06},
    "BRN": {"GDP": 1.4006e10, "GDP_per_capita": 30946.81},
    "ATA": {"GDP": 0, "GDP_per_capita": 0},
    "CYN": {"GDP": 0, "GDP_per_capita": 0},
    "CYP": {"GDP": 2.7835e10, "GDP_per_capita": 22081.10},
    "DJI": {"GDP": 3.483e9, "GDP_per_capita": 3107.56},
    "SOL": {"GDP": 0, "GDP_per_capita": 0},
    "MKD": {"GDP": 1.4052e10, "GDP_per_capita": 6737.92},
    "MNE": {"GDP": 6.102e9, "GDP_per_capita": 9735.54},
    "KOS": {"GDP": 9.412e9, "GDP_per_capita": 5339.45},
    "TTO": {"GDP": 2.5053e10, "GDP_per_capita": 17809.19},
    "SDS": {"GDP": 0, "GDP_per_capita": 0}
}

def get_economy_data():
    cache_file = 'country_economy_cache.csv'
    
    if os.path.exists(cache_file):
        print("Reading economy data from cache...")
        return pd.read_csv(cache_file)
    
    try:
        print("Downloading economy data...")
        # Fetch GDP and GDP per capita data for all countries, most recent year
        data = wb.data.DataFrame(['NY.GDP.MKTP.CD', 'NY.GDP.PCAP.CD'], mrv=1)
        
        # Reset index to convert multi-index to columns
        data = data.reset_index()
        
        # Rename columns
        data.columns = ['code', 'GDP', 'GDP_per_capita']
        
        # Convert GDP and GDP per capita to float, replacing NaN with 0
        data['GDP'] = data['GDP'].fillna(0).astype(float)
        data['GDP_per_capita'] = data['GDP_per_capita'].fillna(0).astype(float)
        
        # Add additional economies
        for code, economy in additional_economies.items():
            if code not in data['code'].values:
                data = data.append({'code': code, 'GDP': economy['GDP'], 'GDP_per_capita': economy['GDP_per_capita']}, ignore_index=True)
            elif data.loc[data['code'] == code, 'GDP'].iloc[0] == 0:
                data.loc[data['code'] == code, 'GDP'] = economy['GDP']
                data.loc[data['code'] == code, 'GDP_per_capita'] = economy['GDP_per_capita']
        
        # Add tier information
        data['tier'] = data['GDP_per_capita'].apply(get_tier)
        
        # Cache the data
        data.to_csv(cache_file, index=False)
        
        print(f"Successfully fetched economy data for {len(data)} countries.")
        return data
    
    except Exception as e:
        print(f"Error downloading or processing data: {e}")
        return pd.DataFrame()

def get_tier(gdp_per_capita):
    if gdp_per_capita >= 20000:
        return 'tier1'
    elif gdp_per_capita >= 5000:
        return 'tier2'
    else:
        return 'tier3'

def create_gdp_mapping(df):
    return dict(zip(df['code'], df['GDP']))

def create_gdp_per_capita_mapping(df):
    return dict(zip(df['code'], zip(df['GDP_per_capita'], df['tier'])))

def create_economy_csv():
    df = get_economy_data()
    if not df.empty:
        # Save all columns to the same CSV file
        df.to_csv('country_economy.csv', index=False)
        
        print("country_economy.csv has been created successfully.")
        print(f"Number of countries in economy data: {len(df)}")
    else:
        print("Failed to create economy CSV file due to data retrieval issues.")

if __name__ == "__main__":
    create_economy_csv()