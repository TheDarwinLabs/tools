import requests
import pandas as pd

def fetch_country_languages():
    url = "https://restcountries.com/v3.1/all?fields=cca3,languages"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        country_languages = []
        for country in data:
            country_code = country['cca3']
            languages = list(country.get('languages', {}).values())
            
            # Apply the logic for official language selection
            if 'English' in languages:
                official_language = 'English'
            elif 'French' in languages:
                official_language = 'French'
            elif languages:
                official_language = languages[0]
            else:
                official_language = 'Unknown'
            
            country_languages.append({'code': country_code, 'language': official_language})

        return pd.DataFrame(country_languages)
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame(columns=['code', 'language'])

# Fetch and save the data
df = fetch_country_languages()

if df.empty:
    print("Error: No data was fetched. Using a sample dataset for demonstration.")
    df = pd.DataFrame({
        'code': ['USA', 'CHN', 'IND'],
        'language': ['English', 'Mandarin Chinese', 'Hindi']
    })

df.to_csv('country_languages.csv', index=False)
print("\nCountry-language data saved to 'country_languages.csv'")

# Function to get official language for a country code
def get_official_language_by_code(country_code):
    if country_code in df['code'].values:
        return df[df['code'] == country_code]['language'].iloc[0]
    return 'Unknown'

# Example usage
print("\nTesting get_official_language_by_code function:")
print("USA:", get_official_language_by_code("USA"))
print("CHN:", get_official_language_by_code("CHN"))
print("Unknown Country:", get_official_language_by_code("XYZ"))

# Print some statistics
print(f"\nTotal number of countries: {len(df)}")
print("First 5 entries:")
print(df.head())