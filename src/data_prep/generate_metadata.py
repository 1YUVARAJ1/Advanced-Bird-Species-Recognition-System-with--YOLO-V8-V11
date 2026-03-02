import os
import pandas as pd
import requests
import time

def get_species_info(species_name):
    """
    Mocking GBIF/IUCN API for now. In a real scenario, this would hit 
    'https://api.gbif.org/v1/species/match?name={species_name}'
    """
    # Simple hardcoded mock logic for demonstration
    species_name_clean = species_name.replace('_', ' ').title()
    if 'Eagle' in species_name_clean or 'Osprey' in species_name_clean:
        return {
            'scientific_name': f'Mock_{species_name_clean.split()[-1]}_sci',
            'family': 'Accipitridae',
            'habitat': 'Wetlands, Rivers',
            'iucn_status': 'Least Concern'
        }
    return {
        'scientific_name': f'Mock_Sci_Name_{species_name_clean}',
        'family': 'Mock_Family',
        'habitat': 'Forests, Woodlands',
        'iucn_status': 'Least Concern'
    }

def generate_metadata(dataset_dir="data/raw/CUB_200_2011", output_file="data/processed/bird_metadata.csv"):
    classes_file = os.path.join(dataset_dir, "classes.txt")
    if not os.path.exists(classes_file):
        print(f"Error: {classes_file} not found.")
        return

    classes = pd.read_csv(classes_file, sep=" ", header=None, names=["class_id", "class_name"])
    
    print("Generating metadata for", len(classes), "species...")
    
    metadata_records = []
    for _, row in classes.iterrows():
        # Remove the preceding number if it exists in class_name from CUB (e.g. '001.Black_footed_Albatross')
        raw_name = row['class_name']
        clean_name = raw_name.split('.')[-1] if '.' in raw_name else raw_name
        
        info = get_species_info(clean_name)
        
        metadata_records.append({
            'species': clean_name.replace('_', ' '),
            'scientific_name': info['scientific_name'],
            'family': info['family'],
            'habitat': info['habitat'],
            'iucn_status': info['iucn_status']
        })
        
    df_meta = pd.DataFrame(metadata_records)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_meta.to_csv(output_file, index=False)
    print(f"Metadata generation complete! Saved to {output_file}")
    
    print("\nSample Data:")
    print(df_meta.head())

if __name__ == "__main__":
    generate_metadata()
