#!/usr/bin/env python3
"""
Docker-compatible dataset repair utility for LLMForge
Run this inside the container to fix corrupted datasets
"""

import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_and_list_datasets():
    """Find all potential dataset directories"""
    # Common dataset paths in LLMForge containers
    search_paths = [
        "/app/datasets",
        "/app/data", 
        "/tmp",
        "/app/storage",
        "/app/tmp"
    ]
    
    dataset_dirs = []
    for search_path in search_paths:
        path = Path(search_path)
        if path.exists():
            logger.info(f"Searching in: {path}")
            try:
                for item in path.rglob("*"):
                    if item.is_dir():
                        # Check if it looks like a dataset directory
                        files = list(item.iterdir()) if item.exists() else []
                        if any(f.suffix in ['.arrow', '.parquet', '.json'] for f in files if f.is_file()):
                            dataset_dirs.append(str(item))
                            logger.info(f"Found potential dataset: {item}")
            except PermissionError:
                logger.warning(f"Permission denied: {search_path}")
            except Exception as e:
                logger.warning(f"Error searching {search_path}: {e}")
    
    return dataset_dirs

def create_repair_commands(dataset_dirs):
    """Create Docker commands to repair datasets"""
    commands = []
    
    # Base command to run inside container
    base_cmd = "docker exec -it"
    
    # Find container name (assuming training service container)
    container_patterns = [
        "training-service",
        "llmforge-training", 
        "training_service",
        "llmforge_training"
    ]
    
    for dataset_dir in dataset_dirs:
        # Create Python repair command
        repair_script = f'''
import sys
import os
sys.path.append("/app")

try:
    from datasets import load_from_disk, Dataset
    import json
    from pathlib import Path
    
    dataset_path = "{dataset_dir}"
    print(f"Testing dataset: {{dataset_path}}")
    
    try:
        dataset = load_from_disk(dataset_path)
        print(f"✅ Dataset OK: {{len(dataset)}} samples")
    except Exception as e:
        print(f"❌ Dataset CORRUPTED: {{e}}")
        
        # Try to repair from JSON files
        path = Path(dataset_path)
        json_files = list(path.glob("*.json"))
        
        if json_files:
            print(f"Found JSON files: {{json_files}}")
            all_data = []
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        else:
                            all_data.append(data)
                except Exception as je:
                    print(f"Failed to read {{json_file}}: {{je}}")
            
            if all_data:
                print(f"Recreating dataset from {{len(all_data)}} samples")
                
                # Backup corrupted dataset
                backup_path = str(path) + "_corrupted_backup"
                if path.exists():
                    import shutil
                    shutil.move(str(path), backup_path)
                    print(f"Backed up corrupted dataset to: {{backup_path}}")
                
                # Create new dataset
                new_dataset = Dataset.from_list(all_data)
                new_dataset.save_to_disk(dataset_path)
                print(f"✅ Dataset repaired: {{len(new_dataset)}} samples")
                
                # Verify
                test_dataset = load_from_disk(dataset_path)
                print(f"✅ Verification successful: {{len(test_dataset)}} samples")
            else:
                print("❌ No valid JSON data found for repair")
        else:
            print("❌ No JSON backup files found")
            
            # Create minimal test dataset as last resort
            print("Creating minimal test dataset...")
            test_data = [
                {{"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1], "labels": [1, 2, 3, 4, 5]}},
                {{"input_ids": [6, 7, 8, 9, 10], "attention_mask": [1, 1, 1, 1, 1], "labels": [6, 7, 8, 9, 10]}}
            ]
            
            # Backup corrupted dataset
            backup_path = str(path) + "_corrupted_backup"
            if path.exists():
                import shutil
                shutil.move(str(path), backup_path)
                
            minimal_dataset = Dataset.from_list(test_data)
            minimal_dataset.save_to_disk(dataset_path)
            print(f"⚠️ Created minimal test dataset: {{len(minimal_dataset)}} samples")
            print("⚠️ Replace with real data when available!")

except ImportError as ie:
    print(f"Import error: {{ie}}")
    print("Make sure this runs inside the container with datasets library")
'''
        
        commands.append(f'python3 -c "{repair_script}"')
    
    return commands

def main():
    logger.info("=== LLMForge Dataset Finder ===")
    
    # Find datasets
    dataset_dirs = find_and_list_datasets()
    
    if not dataset_dirs:
        logger.info("No datasets found in common locations")
        logger.info("Datasets might be in container-specific paths")
        return
    
    logger.info(f"Found {len(dataset_dirs)} potential datasets:")
    for i, dataset_dir in enumerate(dataset_dirs, 1):
        logger.info(f"  {i}. {dataset_dir}")
    
    # Create repair commands
    repair_commands = create_repair_commands(dataset_dirs)
    
    logger.info("\n=== Docker Repair Commands ===")
    logger.info("Run these commands to repair datasets inside the container:")
    logger.info("\n1. First, find your container name:")
    logger.info("   docker ps | grep training")
    logger.info("\n2. Then run the repair commands:")
    
    for i, cmd in enumerate(repair_commands, 1):
        logger.info(f"\n# Repair dataset {i}")
        logger.info(f'docker exec -it <CONTAINER_NAME> {cmd}')
    
    # Also create a combined repair script
    combined_script = '''
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
        print(f"\\n=== Checking {dataset_dir} ===")
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
'''
    
    # Save the combined script
    script_path = Path("repair_all_datasets.py")
    with open(script_path, 'w') as f:
        f.write(combined_script)
    
    logger.info(f"\n3. Or use the combined repair script:")
    logger.info(f"   docker cp {script_path} <CONTAINER_NAME>:/tmp/")
    logger.info(f"   docker exec -it <CONTAINER_NAME> python3 /tmp/{script_path}")

if __name__ == "__main__":
    main()