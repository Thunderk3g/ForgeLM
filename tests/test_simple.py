#!/usr/bin/env python3
"""
Simple test for TrainingArguments parameter compatibility.
"""
import inspect
from transformers import TrainingArguments

def validate_training_args_compatibility():
    """Validate TrainingArguments parameter compatibility with current transformers version."""
    # Get supported parameters for current TrainingArguments
    signature = inspect.signature(TrainingArguments.__init__)
    supported_params = set(signature.parameters.keys()) - {'self'}
    
    return {
        "supported_params": supported_params,
        "has_eval_strategy": "eval_strategy" in supported_params,
        "has_evaluation_strategy": "evaluation_strategy" in supported_params,
        "has_use_gradient_checkpointing": "use_gradient_checkpointing" in supported_params,
        "has_optim": "optim" in supported_params,
    }

def test_parameter_compatibility():
    """Test actual parameter compatibility with current transformers version."""
    print("=== Testing Parameter Compatibility ===")
    
    compatibility_info = validate_training_args_compatibility()
    print(f"Compatibility info: {compatibility_info}")
    
    # Test creating TrainingArguments with both parameter names
    test_args_eval_strategy = {
        "output_dir": "/tmp/test",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "eval_strategy": "no",
    }
    
    test_args_evaluation_strategy = {
        "output_dir": "/tmp/test", 
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "evaluation_strategy": "no",
    }
    
    # Try eval_strategy first (newer parameter)
    try:
        args = TrainingArguments(**test_args_eval_strategy)
        print("✅ 'eval_strategy' parameter works")
        eval_strategy_works = True
    except TypeError as e:
        print(f"❌ 'eval_strategy' parameter failed: {e}")
        eval_strategy_works = False
    
    # Try evaluation_strategy (older parameter)
    try:
        args = TrainingArguments(**test_args_evaluation_strategy)
        print("✅ 'evaluation_strategy' parameter works")
        evaluation_strategy_works = True
    except TypeError as e:
        print(f"❌ 'evaluation_strategy' parameter failed: {e}")
        evaluation_strategy_works = False
    
    print(f"Parameter compatibility: eval_strategy={eval_strategy_works}, evaluation_strategy={evaluation_strategy_works}")
    
    # Test the parameter selection logic from fine_tuning.py
    if compatibility_info["has_evaluation_strategy"]:
        selected_param = "evaluation_strategy"
    elif compatibility_info["has_eval_strategy"]:
        selected_param = "eval_strategy" 
    else:
        selected_param = "eval_strategy"
    
    print(f"Selected parameter based on logic: {selected_param}")
    
    # Test the selected parameter
    test_args = {
        "output_dir": "/tmp/test",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        selected_param: "no",
    }
    
    try:
        args = TrainingArguments(**test_args)
        print(f"✅ Selected parameter '{selected_param}' works correctly")
        return True
    except TypeError as e:
        print(f"❌ Selected parameter '{selected_param}' failed: {e}")
        return False

if __name__ == "__main__":
    success = test_parameter_compatibility()
    if success:
        print("\n🎉 Parameter compatibility test PASSED!")
    else:
        print("\n❌ Parameter compatibility test FAILED!")