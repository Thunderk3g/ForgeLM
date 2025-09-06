#!/usr/bin/env python3
"""
Test script for fine-tuning functionality
Creates a mock scenario to test and debug fine-tuning issues
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datasets import Dataset, DatasetDict

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fine_tuning import fine_tune, FineTuneRequest

def create_mock_dataset():
    """Create a mock dataset for testing."""
    print("Creating mock dataset...")
    
    # Sample training data
    train_data = [
        {
            "instruction": "What is the capital of France?",
            "input": "",
            "output": "The capital of France is Paris."
        },
        {
            "instruction": "Explain photosynthesis",
            "input": "in simple terms",
            "output": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."
        },
        {
            "instruction": "Write a short poem",
            "input": "about nature",
            "output": "Trees sway in gentle breeze,\nBirds sing their melodies,\nNature's beauty brings us peace,\nIn every flower and every leaf."
        },
        {
            "instruction": "Calculate the area of a rectangle",
            "input": "length = 5, width = 3",
            "output": "The area of a rectangle is calculated by multiplying length × width. So 5 × 3 = 15 square units."
        }
    ]
    
    # Sample validation data
    val_data = [
        {
            "instruction": "What is 2 + 2?",
            "input": "",
            "output": "2 + 2 equals 4."
        }
    ]
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset

def create_temp_directories():
    """Create temporary directories for testing."""
    print("Creating temporary directories...")
    
    temp_dir = tempfile.mkdtemp(prefix="llmforge_test_")
    
    # Create directory structure
    dirs = {
        "base": Path(temp_dir) / "models" / "base",
        "output": Path(temp_dir) / "models" / "finetuned", 
        "data": Path(temp_dir) / "data" / "prepared"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create data directories
    (dirs["data"] / "train").mkdir(exist_ok=True)
    (dirs["data"] / "val").mkdir(exist_ok=True)
    
    return temp_dir, dirs

def save_mock_datasets(train_dataset, val_dataset, data_dir):
    """Save mock datasets to disk."""
    print("Saving mock datasets...")
    
    train_path = data_dir / "train"
    val_path = data_dir / "val"
    
    train_dataset.save_to_disk(str(train_path))
    val_dataset.save_to_disk(str(val_path))
    
    print(f"Train dataset saved to: {train_path}")
    print(f"Validation dataset saved to: {val_path}")
    
    # Print dataset info
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Train dataset columns: {train_dataset.column_names}")
    print(f"Sample train record: {train_dataset[0]}")
    
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Val dataset columns: {val_dataset.column_names}")

def test_data_formatting():
    """Test the data formatting functions."""
    print("\n=== Testing Data Formatting ===")
    
    train_dataset, val_dataset = create_mock_dataset()
    
    # Test basic access
    try:
        sample = train_dataset[0]
        instruction = sample["instruction"]
        input_text = sample["input"]
        output_text = sample["output"]
        
        print(f"✅ Data access successful:")
        print(f"  Instruction: {instruction}")
        print(f"  Input: '{input_text}'")
        print(f"  Output: {output_text}")
        
    except Exception as e:
        print(f"❌ Data access failed: {e}")
        return False
    
    # Test batch access
    try:
        batch = train_dataset[:2]
        instructions = batch["instruction"]
        inputs = batch.get("input", [""] * len(instructions))
        outputs = batch["output"]
        
        print(f"✅ Batch access successful:")
        print(f"  Instructions: {len(instructions)} items")
        print(f"  Inputs: {len(inputs)} items") 
        print(f"  Outputs: {len(outputs)} items")
        
    except Exception as e:
        print(f"❌ Batch access failed: {e}")
        return False
    
    return True

def create_mock_model_config(base_dir):
    """Create a mock model configuration."""
    print("Creating mock model directory structure...")
    
    model_dir = base_dir / "unsloth_llama-3.1-8b-unsloth-bnb-4bit"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal config files (these won't be used in real loading but might help with paths)
    config = {
        "model_type": "llama",
        "vocab_size": 32000,
        "hidden_size": 4096,
    }
    
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Mock model config created at: {model_dir}")
    return str(model_dir)

def run_fine_tuning_test():
    """Run the actual fine-tuning test."""
    print("\n=== Running Fine-Tuning Test ===")
    
    # Create temporary directories
    temp_dir, dirs = create_temp_directories()
    
    try:
        # Create and save mock datasets
        train_dataset, val_dataset = create_mock_dataset()
        save_mock_datasets(train_dataset, val_dataset, dirs["data"])
        
        # Create mock model directory
        model_path = create_mock_model_config(dirs["base"])
        
        # Create fine-tuning request
        request = FineTuneRequest(
            base_model_dir=model_path,
            dataset_dir=str(dirs["data"]),
            output_dir=str(dirs["output"]),
            ft_type="instruction",  # This is likely the issue - testing instruction type
            peft_method="qlora",
            alignment="none",
            epochs=1,  # Short test
            batch_size=1,
            learning_rate=1e-4,
            gradient_accumulation_steps=2,
            max_seq_length=256,  # Shorter for testing
        )
        
        print(f"\nFine-tuning request:")
        print(f"  Base model: {request.base_model_dir}")
        print(f"  Dataset: {request.dataset_dir}")
        print(f"  Output: {request.output_dir}")
        print(f"  Type: {request.ft_type}")
        print(f"  Method: {request.peft_method}")
        
        # Run fine-tuning
        print("\nStarting fine-tuning...")
        result = fine_tune(request)
        
        print(f"\nResult: {result}")
        
        if result["status"] == "success":
            print("✅ Fine-tuning test PASSED!")
        else:
            print(f"❌ Fine-tuning test FAILED: {result['message']}")
            
        return result
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
        
    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Cleanup warning: {e}")

def main():
    """Main test function."""
    print("🧪 LLMForge Fine-Tuning Test Script")
    print("=" * 50)
    
    # Test 1: Data formatting
    if not test_data_formatting():
        print("❌ Data formatting test failed. Stopping.")
        return
    
    print("✅ Data formatting test passed!")
    
    # Test 2: Fine-tuning
    result = run_fine_tuning_test()
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"Status: {result.get('status', 'unknown')}")
    if result.get("message"):
        print(f"Message: {result['message']}")
    
    if result.get("status") == "success":
        print("🎉 All tests passed!")
    else:
        print("⚠️  Tests revealed issues that need fixing.")

if __name__ == "__main__":
    main()