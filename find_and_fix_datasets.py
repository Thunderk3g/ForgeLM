#!/usr/bin/env python3
"""
Find and fix all corrupted datasets in the LLMForge system
"""

import os
import json
from pathlib import Path
from datasets import load_from_disk
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_dataset_directories(base_path="/app"):
    """Find all potential dataset directories"""
    base = Path(base_path)
    dataset_dirs = []
    
    # Common dataset paths in LLMForge
    search_paths = [
        base / "datasets",
        base / "data", 
        base / "tmp",
        base / "storage",
        Path("/tmp"),
        Path("/app/data"),
        Path("/app/datasets"),
        Path("/app/storage")
    ]
    
    for search_path in search_paths:
        if search_path.exists():
            logger.info(f"Searching in: {search_path}")
            try:
                for item in search_path.rglob("*"):
                    if item.is_dir():
                        # Check if it looks like a dataset directory
                        if any(f.suffix in ['.arrow', '.parquet'] for f in item.iterdir() if f.is_file()):
                            dataset_dirs.append(item)
                            logger.info(f"Found potential dataset: {item}")
            except PermissionError:
                logger.warning(f"Permission denied accessing: {search_path}")
            except Exception as e:
                logger.warning(f"Error searching {search_path}: {e}")
    
    return dataset_dirs

def test_dataset_loading(dataset_path):
    """Test if a dataset can be loaded"""
    try:
        dataset = load_from_disk(str(dataset_path))
        logger.info(f"✅ Dataset OK: {dataset_path} ({len(dataset)} samples)")
        return True
    except Exception as e:
        logger.error(f"❌ Dataset CORRUPTED: {dataset_path} - Error: {e}")
        return False

def fix_corrupted_dataset(dataset_path):
    """Fix a corrupted dataset using the repair utility"""
    logger.info(f"Attempting to fix: {dataset_path}")
    
    # Import the repair utility
    import sys
    sys.path.append(str(Path(__file__).parent))
    from fix_dataset_schema import DatasetSchemaRepairer
    
    repairer = DatasetSchemaRepairer(str(dataset_path))
    
    # Diagnose first
    diagnosis = repairer.diagnose_dataset()
    logger.info(f"Diagnosis for {dataset_path}:")
    logger.info(json.dumps(diagnosis, indent=2))
    
    # Attempt repair
    if repairer.repair_dataset():
        logger.info(f"✅ Successfully repaired: {dataset_path}")
        return True
    else:
        logger.error(f"❌ Failed to repair: {dataset_path}")
        return False

def main():
    logger.info("=== LLMForge Dataset Repair Utility ===")
    
    # Find all datasets
    logger.info("Searching for datasets...")
    dataset_dirs = find_dataset_directories()
    
    if not dataset_dirs:
        logger.warning("No dataset directories found!")
        return
        
    logger.info(f"Found {len(dataset_dirs)} potential datasets")
    
    # Test each dataset
    corrupted_datasets = []
    for dataset_dir in dataset_dirs:
        if not test_dataset_loading(dataset_dir):
            corrupted_datasets.append(dataset_dir)
    
    if not corrupted_datasets:
        logger.info("🎉 All datasets are healthy!")
        return
        
    logger.warning(f"Found {len(corrupted_datasets)} corrupted datasets")
    
    # Fix corrupted datasets
    for corrupted_dataset in corrupted_datasets:
        logger.info(f"\n=== Fixing {corrupted_dataset} ===")
        if fix_corrupted_dataset(corrupted_dataset):
            # Verify fix
            if test_dataset_loading(corrupted_dataset):
                logger.info(f"✅ Repair verified for: {corrupted_dataset}")
            else:
                logger.error(f"❌ Repair verification failed for: {corrupted_dataset}")
        else:
            logger.error(f"❌ Could not repair: {corrupted_dataset}")
            
    logger.info("\n=== Repair Summary ===")
    logger.info("Re-testing all datasets...")
    
    still_corrupted = []
    for dataset_dir in dataset_dirs:
        if not test_dataset_loading(dataset_dir):
            still_corrupted.append(dataset_dir)
            
    if still_corrupted:
        logger.error(f"❌ {len(still_corrupted)} datasets still corrupted:")
        for dataset in still_corrupted:
            logger.error(f"   - {dataset}")
    else:
        logger.info("🎉 All datasets now healthy!")

if __name__ == "__main__":
    main()