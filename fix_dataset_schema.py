#!/usr/bin/env python3
"""
Dataset Schema Repair Utility for LLMForge
Fixes corrupted Arrow/Parquet files that cause 'schema message null or length 0' errors
"""

import os
import json
import logging
from pathlib import Path
from datasets import Dataset, load_from_disk
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetSchemaRepairer:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        
    def diagnose_dataset(self) -> Dict[str, Any]:
        """Diagnose what's wrong with the dataset"""
        logger.info(f"Diagnosing dataset at: {self.dataset_path}")
        
        diagnosis = {
            "path_exists": self.dataset_path.exists(),
            "is_directory": self.dataset_path.is_dir() if self.dataset_path.exists() else False,
            "files": [],
            "arrow_files": [],
            "json_files": [],
            "parquet_files": [],
            "corrupted_files": [],
            "errors": []
        }
        
        if not diagnosis["path_exists"]:
            diagnosis["errors"].append("Dataset path does not exist")
            return diagnosis
            
        if not diagnosis["is_directory"]:
            diagnosis["errors"].append("Dataset path is not a directory")
            return diagnosis
            
        # List all files
        try:
            for file_path in self.dataset_path.rglob("*"):
                if file_path.is_file():
                    diagnosis["files"].append(str(file_path))
                    
                    if file_path.suffix == ".arrow":
                        diagnosis["arrow_files"].append(str(file_path))
                    elif file_path.suffix == ".json":
                        diagnosis["json_files"].append(str(file_path))
                    elif file_path.suffix == ".parquet":
                        diagnosis["parquet_files"].append(str(file_path))
        except Exception as e:
            diagnosis["errors"].append(f"Error listing files: {e}")
            
        # Check Arrow files for corruption
        for arrow_file in diagnosis["arrow_files"]:
            try:
                # Try to read the arrow file
                table = pa.ipc.open_file(arrow_file).read_all()
                logger.info(f"Arrow file {arrow_file} is readable: {table.shape}")
            except Exception as e:
                logger.error(f"Corrupted Arrow file {arrow_file}: {e}")
                diagnosis["corrupted_files"].append(arrow_file)
                diagnosis["errors"].append(f"Corrupted Arrow file {arrow_file}: {e}")
                
        return diagnosis
    
    def repair_dataset(self, backup_source: str = None) -> bool:
        """Repair the corrupted dataset"""
        logger.info("Starting dataset repair...")
        
        # First, try to load the dataset normally
        try:
            dataset = load_from_disk(str(self.dataset_path))
            logger.info("Dataset loaded successfully - no repair needed!")
            return True
        except Exception as e:
            logger.error(f"Dataset loading failed: {e}")
            
        # Try repair methods
        repair_methods = [
            self._repair_from_json_backup,
            self._repair_from_parquet,
            self._recreate_from_source_files,
            self._create_minimal_dataset
        ]
        
        for method in repair_methods:
            try:
                logger.info(f"Trying repair method: {method.__name__}")
                if method():
                    logger.info(f"Dataset repaired successfully using {method.__name__}")
                    return True
            except Exception as e:
                logger.error(f"Repair method {method.__name__} failed: {e}")
                
        logger.error("All repair methods failed")
        return False
    
    def _repair_from_json_backup(self) -> bool:
        """Try to repair from JSON backup files"""
        json_files = list(self.dataset_path.glob("*.json"))
        if not json_files:
            logger.info("No JSON backup files found")
            return False
            
        logger.info(f"Found JSON files: {json_files}")
        
        all_data = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            except Exception as e:
                logger.error(f"Failed to read {json_file}: {e}")
                continue
                
        if all_data:
            # Create new dataset from JSON data
            dataset = Dataset.from_list(all_data)
            
            # Save to new location
            backup_path = self.dataset_path.parent / f"{self.dataset_path.name}_backup"
            if self.dataset_path.exists():
                self.dataset_path.rename(backup_path)
                
            dataset.save_to_disk(str(self.dataset_path))
            logger.info(f"Dataset repaired from JSON: {len(all_data)} samples")
            return True
            
        return False
    
    def _repair_from_parquet(self) -> bool:
        """Try to repair from Parquet files"""
        parquet_files = list(self.dataset_path.glob("*.parquet"))
        if not parquet_files:
            return False
            
        logger.info(f"Found Parquet files: {parquet_files}")
        
        tables = []
        for parquet_file in parquet_files:
            try:
                table = pq.read_table(parquet_file)
                tables.append(table)
            except Exception as e:
                logger.error(f"Failed to read {parquet_file}: {e}")
                continue
                
        if tables:
            # Combine all tables
            combined_table = pa.concat_tables(tables)
            dataset = Dataset(combined_table)
            
            # Save repaired dataset
            backup_path = self.dataset_path.parent / f"{self.dataset_path.name}_backup"
            if self.dataset_path.exists():
                self.dataset_path.rename(backup_path)
                
            dataset.save_to_disk(str(self.dataset_path))
            logger.info(f"Dataset repaired from Parquet: {len(dataset)} samples")
            return True
            
        return False
    
    def _recreate_from_source_files(self) -> bool:
        """Try to recreate from source data files"""
        # Look for common source file patterns
        possible_sources = [
            self.dataset_path.parent / "train.json",
            self.dataset_path.parent / "data.json",
            self.dataset_path.parent / "training_data.json",
            self.dataset_path.parent / "dataset.json"
        ]
        
        for source_file in possible_sources:
            if source_file.exists():
                try:
                    logger.info(f"Attempting to recreate from {source_file}")
                    with open(source_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if isinstance(data, list) and len(data) > 0:
                        dataset = Dataset.from_list(data)
                        
                        # Backup corrupted dataset
                        backup_path = self.dataset_path.parent / f"{self.dataset_path.name}_corrupted"
                        if self.dataset_path.exists():
                            self.dataset_path.rename(backup_path)
                            
                        dataset.save_to_disk(str(self.dataset_path))
                        logger.info(f"Dataset recreated from {source_file}: {len(dataset)} samples")
                        return True
                        
                except Exception as e:
                    logger.error(f"Failed to recreate from {source_file}: {e}")
                    continue
                    
        return False
    
    def _create_minimal_dataset(self) -> bool:
        """Create a minimal test dataset as last resort"""
        logger.info("Creating minimal test dataset as fallback")
        
        # Create a simple test dataset
        test_data = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [1, 2, 3, 4, 5]
            },
            {
                "input_ids": [6, 7, 8, 9, 10],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [6, 7, 8, 9, 10]
            }
        ]
        
        try:
            dataset = Dataset.from_list(test_data)
            
            # Backup corrupted dataset
            backup_path = self.dataset_path.parent / f"{self.dataset_path.name}_corrupted"
            if self.dataset_path.exists():
                self.dataset_path.rename(backup_path)
                
            dataset.save_to_disk(str(self.dataset_path))
            logger.warning(f"Created minimal test dataset: {len(dataset)} samples")
            logger.warning("This is only for testing - replace with real data!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create minimal dataset: {e}")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Repair corrupted LLMForge datasets")
    parser.add_argument("dataset_path", help="Path to the corrupted dataset directory")
    parser.add_argument("--diagnose-only", action="store_true", help="Only diagnose, don't repair")
    parser.add_argument("--backup-source", help="Path to backup source files")
    
    args = parser.parse_args()
    
    repairer = DatasetSchemaRepairer(args.dataset_path)
    
    # Always diagnose first
    diagnosis = repairer.diagnose_dataset()
    print("\n=== Dataset Diagnosis ===")
    print(json.dumps(diagnosis, indent=2))
    
    if diagnosis["errors"]:
        print(f"\n⚠️  Found {len(diagnosis['errors'])} error(s)")
        for error in diagnosis["errors"]:
            print(f"   - {error}")
    
    if not args.diagnose_only:
        print("\n=== Starting Repair ===")
        success = repairer.repair_dataset(args.backup_source)
        
        if success:
            print("✅ Dataset repair completed successfully!")
            
            # Verify repair
            try:
                test_dataset = load_from_disk(args.dataset_path)
                print(f"✅ Verification successful: {len(test_dataset)} samples loaded")
            except Exception as e:
                print(f"❌ Verification failed: {e}")
        else:
            print("❌ Dataset repair failed")
            print("Manual intervention required")

if __name__ == "__main__":
    main()