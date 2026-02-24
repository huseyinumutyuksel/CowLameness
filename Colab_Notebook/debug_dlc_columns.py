import pandas as pd
import os

# Path to the specific CSV file that was used in the test
csv_path = os.path.join('..', 'DeepLabCutOutputs', 'outputs', 'Saglikli', 'cow_0001DLC_dlcrnetms5_CowGaitAnalysisJan6shuffle1_700000.csv')

print(f"Reading: {csv_path}")

try:
    # DLC CSVs usually have a 3-level header (scorer, bodyparts, coords)
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    
    print("\n--- Column Structure ---")
    # Print the first few columns to see the structure
    print(df.columns[:5])
    
    print("\n--- Extracted Bodyparts ---")
    # Extract unique bodyparts from the second level of the header
    bodyparts = set()
    for col in df.columns:
        # col is a tuple: (scorer, bodypart, coord)
        if len(col) >= 2:
            bodyparts.add(col[1])
            
    for bp in sorted(bodyparts):
        print(f"  {bp}")
        
except Exception as e:
    print(f"Error reading CSV: {e}")
