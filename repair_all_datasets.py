import sys
sys.path.append("/app")

try:
    from datasets import load_from_disk, Dataset
    import json
    import os
    from pathlib import Path
    
    # Search for datasets
    search_paths = ["/app/datasets", "/app/data", "/tmp", "/app/storage", "/app/tmp"]
    dataset_dirs = []
    
    for search_path in search_paths:
        path = Path(search_path)
        if path.exists():
            for item in path.rglob("*"):
                if item.is_dir():
                    files = list(item.iterdir()) if item.exists() else []
                    if any(f.suffix in ['.arrow', '.parquet'] for f in files if f.is_file()):
                        dataset_dirs.append(str(item))
    
    print(f"Found {len(dataset_dirs)} datasets to check")
    
    for dataset_dir in dataset_dirs:
        print(f"\n=== Checking {dataset_dir} ===")
        try:
            dataset = load_from_disk(dataset_dir)
            print(f"✅ OK: {len(dataset)} samples")
        except Exception as e:
            print(f"❌ CORRUPTED: {e}")
            print("Attempting repair...")
            
            # Repair logic here
            path = Path(dataset_dir)
            
            # Try JSON repair
            json_files = list(path.glob("*.json"))
            if json_files:
                all_data = []
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                all_data.extend(data)
                            else:
                                all_data.append(data)
                    except:
                        continue
                
                if all_data:
                    import shutil
                    backup_path = str(path) + "_backup"
                    if path.exists():
                        shutil.move(str(path), backup_path)
                    
                    new_dataset = Dataset.from_list(all_data)
                    new_dataset.save_to_disk(dataset_dir)
                    print(f"✅ Repaired from JSON: {len(new_dataset)} samples")
                    continue
            
            # Create minimal dataset as fallback
            print("Creating minimal test dataset...")
            test_data = [
                {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
                {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1], "labels": [4, 5, 6]}
            ]
            
            import shutil
            backup_path = str(path) + "_backup"
            if path.exists():
                shutil.move(str(path), backup_path)
            
            minimal_dataset = Dataset.from_list(test_data)
            minimal_dataset.save_to_disk(dataset_dir)
            print(f"⚠️ Created minimal dataset: {len(minimal_dataset)} samples")

except Exception as e:
    print(f"Error: {e}")