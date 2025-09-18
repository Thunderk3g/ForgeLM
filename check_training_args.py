#!/usr/bin/env python3
"""
Check TrainingArguments parameters to identify any invalid ones
"""

import sys
from pathlib import Path

# Add source to path
sys.path.append(str(Path(__file__).parent / "src" / "microservices" / "training_service" / "src"))

def check_training_args_parameters():
    """Check if our parameters are valid for TrainingArguments"""
    print("🔍 Checking TrainingArguments parameters...")
    
    try:
        from transformers import TrainingArguments
        import inspect
        
        # Get TrainingArguments __init__ signature
        sig = inspect.signature(TrainingArguments.__init__)
        valid_params = set(sig.parameters.keys())
        valid_params.remove('self')  # Remove self parameter
        
        print(f"✅ Found {len(valid_params)} valid TrainingArguments parameters")
        
        # Parameters we're trying to use
        our_params = {
            "per_device_train_batch_size",
            "gradient_accumulation_steps", 
            "warmup_steps",
            "num_train_epochs",
            "learning_rate",
            "fp16",
            "bf16",
            "logging_steps",
            "optim",
            "weight_decay",
            "lr_scheduler_type", 
            "seed",
            "output_dir",
            "save_steps",
            "save_total_limit",
            "eval_strategy",
            "dataloader_num_workers",
            "dataloader_pin_memory",
            "remove_unused_columns",
            "run_name",
            "report_to",
            "gradient_checkpointing",
            "max_grad_norm",
            "group_by_length",
            "prediction_loss_only",
        }
        
        # Check which parameters are invalid
        invalid_params = our_params - valid_params
        valid_our_params = our_params & valid_params
        
        print(f"✅ Valid parameters we're using: {len(valid_our_params)}")
        for param in sorted(valid_our_params):
            print(f"   ✓ {param}")
            
        if invalid_params:
            print(f"❌ Invalid parameters we're trying to use: {len(invalid_params)}")
            for param in sorted(invalid_params):
                print(f"   ✗ {param}")
        else:
            print("✅ All our parameters are valid!")
            
        return len(invalid_params) == 0
        
    except Exception as e:
        print(f"❌ Error checking parameters: {e}")
        return False

def main():
    """Check TrainingArguments compatibility"""
    print("🚀 TrainingArguments Parameter Validation\n")
    
    if check_training_args_parameters():
        print("\n🎉 All parameters are valid for TrainingArguments!")
        print("✅ This should fix the 'unexpected keyword argument' error")
        return True
    else:
        print("\n❌ Some parameters are invalid and need to be removed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)