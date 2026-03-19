import os
import json
import pandas as pd
from pathlib import Path

def flatten_complex_types(df):
    """
    Analyzes the DataFrame and converts nested structures (dicts, lists)
    into JSON strings so they can be easily serialized to Parquet.
    """
    for col in df.columns:
        # Check if any element in the column is a dict or a list
        has_complex = df[col].apply(lambda x: isinstance(x, (dict, list))).any()
        if has_complex:
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
            # Ensure the column type is string to avoid mixed types in Parquet
            df[col] = df[col].astype(str)
    return df

def convert_json_to_parquet(json_file_path, output_folder):
    """
    Analyzes the structure of a JSON file and converts it into a Parquet file.
    
    Args:
        json_file_path (str or Path): Path to the input JSON file.
        output_folder (str or Path): Path to the directory where Parquet files will be saved.
    """
    json_path = Path(json_file_path)
    out_dir = Path(output_folder)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # dataset_infos.json usually has {"config_name": {"desc": "...", ...}, ...}
    if isinstance(data, dict):
        # Convert dictionary to DataFrame treating keys as rows
        df = pd.DataFrame.from_dict(data, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'config_name_key'}, inplace=True)
    elif isinstance(data, list):
        # Already a list of records
        df = pd.DataFrame(data)
    else:
        # Fallback for simple values
        df = pd.DataFrame([{'value': data}])
        
    # Flatten dicts and lists to string to be Parquet compatible
    df = flatten_complex_types(df)
    
    # Create the output directory if it does not exist
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract the name of the dataset from the parent directory
    dataset_name = json_path.parent.name
    # Handle the case where the JSON might be at the root of the search directory
    if dataset_name == 'datasets':
        dataset_name = json_path.stem
        
    parquet_filename = f"{dataset_name}_{json_path.stem}.parquet"
    out_file = out_dir / parquet_filename
    
    try:
        df.to_parquet(out_file, engine='pyarrow', index=False)
        print(f"Successfully converted: {json_path} -> {out_file}")
    except Exception as e:
        print(f"Failed to convert {json_path}. Error: {e}")

def main():
    # Base configuration
    base_dir = Path("/mnt/games/projetos/tcc")
    datasets_dir = base_dir / "PoETaV2/lm_eval/datasets"
    output_dir = base_dir / "dataset/poetav2"
    
    print(f"Searching for .json files in: {datasets_dir}")
    print(f"Output directory: {output_dir}")
    
    # Ensure datasets_dir exists
    if not datasets_dir.exists():
        print(f"Error: Directory {datasets_dir} does not exist.")
        return

    # Navigate and process all .json files in the specified directory
    json_files = list(datasets_dir.rglob("*.json"))
    
    if not json_files:
        print("No .json files found in the specified directory.")
        return
        
    for json_file in json_files:
        convert_json_to_parquet(json_file, output_dir)

if __name__ == "__main__":
    main()
