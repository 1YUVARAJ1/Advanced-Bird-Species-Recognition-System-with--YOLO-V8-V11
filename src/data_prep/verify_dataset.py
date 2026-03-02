import os
import pandas as pd

def verify_dataset(dataset_dir="data/raw/CUB_200_2011"):
    classes_file = os.path.join(dataset_dir, "classes.txt")
    if not os.path.exists(classes_file):
        print(f"Error: {classes_file} not found. Ensure dataset is downloaded correctly.")
        return

    classes = pd.read_csv(classes_file, sep=" ", header=None, names=["class_id", "class_name"])
    print(f"Total classes found: {len(classes)}")
    
    eagles = classes[classes['class_name'].str.contains('Eagle', case=False)]
    ospreys = classes[classes['class_name'].str.contains('Osprey', case=False)]
    
    print("\n--- Eagles found ---")
    print(eagles if not eagles.empty else "No Eagles found.")
    
    print("\n--- Ospreys found ---")
    print(ospreys if not ospreys.empty else "No Ospreys found.")
    
    print("\nFirst 5 classes in dataset:")
    print(classes.head())

if __name__ == "__main__":
    verify_dataset()
