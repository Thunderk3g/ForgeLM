#!/usr/bin/env python3
"""Test model discovery functionality"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import AppConfig
from models.entities import ModelPair

# Create simple settings
settings = AppConfig()

async def test_discovery():
    """Test basic model discovery logic"""
    from pathlib import Path
    
    pairs = []
    
    # Scan for base models
    base_models_dir = Path(settings.models_base_dir)
    print(f"Scanning base models in: {base_models_dir}")
    
    if base_models_dir.exists():
        for base_dir in base_models_dir.iterdir():
            if base_dir.is_dir():
                base_model_name = base_dir.name
                base_model_path = str(base_dir)
                print(f"Found base model: {base_model_name}")
                
                # Look for corresponding LoRA adapters
                finetuned_dir = Path(settings.models_finetuned_dir)
                print(f"Scanning for adapters in: {finetuned_dir}")
                
                if finetuned_dir.exists():
                    # Main adapter
                    main_adapter = finetuned_dir / "adapter_model.safetensors"
                    if main_adapter.exists():
                        adapter_size_mb = main_adapter.stat().st_size / (1024 * 1024)
                        
                        pair = ModelPair(
                            pair_id=f"{base_model_name}_main",
                            base_model_name=base_model_name,
                            base_model_path=base_model_path,
                            adapter_name="main",
                            adapter_path=str(finetuned_dir),
                            base_model_size_mb=1000.0,  # Placeholder
                            adapter_size_mb=adapter_size_mb
                        )
                        pairs.append(pair)
                        print(f"✓ Created pair: {pair.pair_id}")
                    
                    # Checkpoint adapters
                    checkpoints_dir = finetuned_dir / "checkpoints"
                    if checkpoints_dir.exists():
                        for checkpoint_dir in checkpoints_dir.iterdir():
                            if checkpoint_dir.is_dir() and (checkpoint_dir / "adapter_model.safetensors").exists():
                                checkpoint_name = checkpoint_dir.name
                                adapter_size_mb = (checkpoint_dir / "adapter_model.safetensors").stat().st_size / (1024 * 1024)
                                
                                pair = ModelPair(
                                    pair_id=f"{base_model_name}_{checkpoint_name}",
                                    base_model_name=base_model_name,
                                    base_model_path=base_model_path,
                                    adapter_name=checkpoint_name,
                                    adapter_path=str(checkpoint_dir),
                                    base_model_size_mb=1000.0,  # Placeholder
                                    adapter_size_mb=adapter_size_mb
                                )
                                pairs.append(pair)
                                print(f"✓ Created checkpoint pair: {pair.pair_id}")
    else:
        print(f"❌ Base models directory does not exist: {base_models_dir}")
    
    print(f"\n📊 Total discovered pairs: {len(pairs)}")
    for pair in pairs:
        print(f"  - {pair.pair_id}: {pair.base_model_name} + {pair.adapter_name}")

if __name__ == "__main__":
    asyncio.run(test_discovery())