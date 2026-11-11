"""
PEFT method names and their mapping onto Hugging Face ``peft``.

This module used to carry hand-written DoRA, OFT and PiSSA layers. They were
never called by the training path, and ``peft`` ships maintained
implementations of all three (``use_dora``, ``OFTConfig``,
``init_lora_weights="pissa"``). The enum and the config dataclass are kept for
the YAML config loader; ``PEFTConfig.to_hf()`` builds the real ``peft`` config.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn


class PEFTMethod(Enum):
    """Supported PEFT methods"""
    LORA = "lora"
    QLORA = "qlora"
    DORA = "dora"
    OFT = "oft"
    PISSA = "pissa"
    ADALORA = "adalora"
    FOURIERFT = "fourierft"
    

@dataclass
class PEFTConfig:
    """Configuration for PEFT methods"""
    method: PEFTMethod
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: str = "none"  # none, all, lora_only
    
    # Method-specific parameters
    # DoRA
    use_dora: bool = False
    
    # OFT
    oft_block_size: int = 4
    oft_dropout: float = 0.0
    
    # PiSSA
    pissa_init_method: str = "svd"  # svd, random
    pissa_iter: int = 16
    
    # AdaLoRA
    adalora_target_r: int = 8
    adalora_init_r: int = 12
    adalora_tinit: int = 0
    adalora_tfinal: int = 1000
    adalora_deltaT: int = 10
    adalora_beta1: float = 0.85
    adalora_beta2: float = 0.85
    adalora_orth_reg_weight: float = 0.5


def to_hf_peft_config(config: PEFTConfig):
    """Build the Hugging Face ``peft`` config object for ``config``."""
    from types import SimpleNamespace

    from utils.factory_bridge import resolve_peft_config

    method = "dora" if config.use_dora else config.method.value
    return resolve_peft_config(SimpleNamespace(
        peft_method=method, peft_rank=config.rank, peft_alpha=config.alpha,
        peft_dropout=config.dropout, target_modules=config.target_modules))


def apply_peft_to_model(model: nn.Module, config: PEFTConfig) -> nn.Module:
    """Wrap ``model`` with the adapter described by ``config`` (via ``peft``)."""
    from peft import get_peft_model

    return get_peft_model(model, to_hf_peft_config(config))


def get_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    """Get count of trainable parameters by type"""
    trainable_params = 0
    total_params = 0
    peft_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        if param.requires_grad:
            trainable_params += param.numel()
            
            # Check if it's a PEFT parameter
            if any(peft_name in name.lower() for peft_name in ['lora', 'oft', 'dora', 'pissa', 'magnitude']):
                peft_params += param.numel()
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "peft_params": peft_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0
    }