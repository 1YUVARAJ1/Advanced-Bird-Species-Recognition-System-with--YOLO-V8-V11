import os
import pandas as pd
import requests
import time
import argparse
from tqdm import tqdm

def fetch_gbif_data(common_name):
    """ Fetch Scientific Name and Family from GBIF """
    try:
        # Use fuzzy search and enforce Aves (Bird) classKey=212
        url = f"https://api.gbif.org/v1/species/search?q={common_name}&rank=SPECIES&classKey=212"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                best_match = data['results'][0]
                sci_name = best_match.get('scientificName', 'Unknown')
                if sci_name != 'Unknown' and ' ' in sci_name:
                    sci_name = ' '.join(sci_name.split(' ')[:2])
                return {
                    'scientific_name': sci_name,
                    'family': best_match.get('family', 'Unknown')
                }
    except Exception as e:
        print(f"GBIF Error for {common_name}: {e}")
        
    return {'scientific_name': 'Unknown', 'family': 'Unknown'}

def fetch_wikipedia_habitat(search_term):
    """ Fetch the intro extract from Wikipedia using fuzzy search """
    try:
        # Use fuzzy search (generator=search) instead of exact titles
        url = f"https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch={search_term} bird&gsrlimit=1&prop=extracts&exintro&explaintext&format=json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            if pages:
                # Get the first (and only) search result page
                page_info = list(pages.values())[0]
                extract = page_info.get('extract', '')
                if extract:
                    sentences = extract.split('. ')
                    if len(sentences) >= 2:
                        return f"{sentences[0]}. {sentences[1]}."
                    elif len(sentences) == 1 and sentences[0]:
                        return f"{sentences[0]}."
    except Exception as e:
        pass
    
    return "Various natural habitats (Information unavailable)."

def fetch_iucn_data(scientific_name, token):
    """ Fetch Conservation Status from IUCN if token is provided """
    if not token or token == "YOUR_IUCN_TOKEN_HERE":
        return "Unknown (Requires IUCN Token)"
        
    try:
        # We need just the two-word genus and species for IUCN
        clean_sci_name = ' '.join(scientific_name.split(' ')[:2]).replace(' ', '%20')
        url = f"https://apiv3.iucnredlist.org/api/v3/species/{clean_sci_name}?token={token}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'result' in data and len(data['result']) > 0:
                category = data['result'][0].get('category', 'Unknown')
                
                # Map IUCN codes to full text
                iucn_map = {
                    'LC': 'Least Concern',
                    'NT': 'Near Threatened',
                    'VU': 'Vulnerable',
                    'EN': 'Endangered',
                    'CR': 'Critically Endangered',
                    'EW': 'Extinct in the Wild',
                    'EX': 'Extinct',
                    'DD': 'Data Deficient'
                }
                return iucn_map.get(category, category)
    except Exception as e:
        pass
        
    return "Data Deficient"

def generate_real_metadata(dataset_dir="data/raw/CUB_200_2011", output_file="data/processed/bird_metadata.csv", iucn_token=None):
    classes_file = os.path.join(dataset_dir, "classes.txt")
    if not os.path.exists(classes_file):
        print(f"Error: {classes_file} not found.")
        return

    classes = pd.read_csv(classes_file, sep=" ", header=None, names=["class_id", "class_name"])
    print(f"Fetching real Gbif & Wikipedia API metadata for {len(classes)} species...")
    print(f"This will take several minutes to respect API rate limits...\n")
    
    metadata_records = []
    
    for _, row in tqdm(classes.iterrows(), total=len(classes), desc="Querying APIs"):
        raw_name = row['class_name']
        clean_name = raw_name.split('.')[-1] if '.' in raw_name else raw_name
        common_name = clean_name.replace('_', ' ')
        
        # 1. GBIF - Taxonomy
        gbif_data = fetch_gbif_data(common_name)
        sci_name = gbif_data['scientific_name']
        
        # 2. Wikipedia - Habitat / Description
        # Try scientific name first, fallback to common name if it fails
        habitat_desc = "Various natural habitats (Information unavailable)."
        if sci_name != 'Unknown':
            habitat_desc = fetch_wikipedia_habitat(sci_name)
            
        if habitat_desc == "Various natural habitats (Information unavailable).":
            habitat_desc = fetch_wikipedia_habitat(common_name)
            
        # 3. IUCN - Conservation Status
        iucn_status = fetch_iucn_data(sci_name, iucn_token)
        
        metadata_records.append({
            'species': common_name,
            'scientific_name': sci_name,
            'family': gbif_data['family'],
            'habitat': habitat_desc,
            'iucn_status': iucn_status
        })
        
        # Rate Limiting: Sleep heavily to avoid getting IP banned from Wiki/GBIF
        time.sleep(1.0)
        
    df_meta = pd.DataFrame(metadata_records)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_meta.to_csv(output_file, index=False)
    
    print(f"\n✅ Real Metadata successfully generated and saved to {output_file}!")
    print("\nSample Data:")
    print(df_meta.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch real bird metadata from GBIF, Wikipedia, and IUCN.")
    parser.add_argument("--iucn-token", type=str, default="", help="Your IUCN Red List API Token (optional).")
    args = parser.parse_args()
    
    generate_real_metadata(iucn_token=args.iucn_token)
