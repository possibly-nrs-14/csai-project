import json
import pandas as pd
from pathlib import Path

def compile_results():
    # Define the base directory (the directory where this script is running)
    base_dir = Path('.')
    
    # Use glob to find all pearson_r.json files matching the known folder structure
    # This will automatically pick up sub-01, sub-02, sub-03, sub-05, etc.
    json_files = list(base_dir.glob('sub-*/ridge/*/pearson_r.json'))
    
    if not json_files:
        print("No pearson_r.json files found. Make sure the script is in the parent directory.")
        return

    data = []
    
    # Loop through each found JSON file
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                
                # Append the extracted data as a dictionary
                data.append({
                    'Subject': result.get('subject'),
                    'Feature Key': result.get('feature_key'),
                    'Mean r': result.get('mean_r'),
                    'Median r': result.get('median_r'),
                    'Max r': result.get('max_r'),
                    '% Positive Parcels': result.get('pct_positive')
                })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Convert the list of dictionaries into a Pandas DataFrame
    df = pd.DataFrame(data)
    
    # Sort the table alphabetically by Subject, then by Feature Key
    df = df.sort_values(by=['Subject', 'Feature Key']).reset_index(drop=True)
    
    # Format floating point numbers to 4 decimal places for a cleaner display
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    # Print the table to the console
    print("\n" + "="*70)
    print(" Algonauts 2025: Ridge Regression Results Summary")
    print("="*70)
    print(df.to_markdown(index=False))
    print("="*70 + "\n")
    
    # Save the dataframe to a CSV file in the current directory
    output_file = 'ridge_results_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"✅ Data successfully compiled and saved to: {output_file}")

if __name__ == "__main__":
    compile_results()