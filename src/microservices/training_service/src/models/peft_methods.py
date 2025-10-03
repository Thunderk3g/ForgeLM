"""
Advanced PEFT Methods Implementation

Implements advanced Parameter-Efficient Fine-Tuning methods
inspired by LLaMA-Factory:
- OFT (Orthogonal Fine-Tuning)  
- DoRA (Weight-Decomposed Low-Rank Adaptation)
- PiSSA (Principal Singular values and Singular vectors Adaptation)
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


class DoRALinear(nn.Module):
    """DoRA (Weight-Decomposed Low-Rank Adaptation) implementation"""
    
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze base layer
        self.base_layer.requires_grad_(False)
        
        # LoRA matrices
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)
        
        # DoRA magnitude vector
        self.magnitude = nn.Parameter(torch.ones(base_layer.out_features))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output
        base_output = self.base_layer(x)
        
        # LoRA adaptation
        lora_output = self.lora_B(self.lora_A(x))
        if self.dropout:
            lora_output = self.dropout(lora_output)
        
        # DoRA decomposition
        combined_weight = self.base_layer.weight + self.scaling * self.lora_B.weight @ self.lora_A.weight
        weight_norm = torch.norm(combined_weight, dim=1, keepdim=True)
        
        # Apply magnitude scaling
        magnitude_scaling = self.magnitude.unsqueeze(0) / weight_norm.squeeze()
        
        return base_output * magnitude_scaling.unsqueeze(0) + self.scaling * lora_output


class OFTLinear(nn.Module):
    """OFT (Orthogonal Fine-Tuning) implementation"""
    
    def __init__(self, base_layer: nn.Linear, block_size: int = 4, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.block_size = block_size
        
        # Freeze base layer
        self.base_layer.requires_grad_(False)
        
        # Orthogonal blocks
        out_features = base_layer.out_features
        self.num_blocks = out_features // block_size
        
        # Orthogonal parameters (skew-symmetric matrices)
        self.oft_blocks = nn.Parameter(
            torch.zeros(self.num_blocks, block_size, block_size)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize OFT parameters"""
        nn.init.zeros_(self.oft_blocks)
    
    def get_orthogonal_matrix(self) -> torch.Tensor:
        """Compute orthogonal transformation matrix"""
        # Create block diagonal orthogonal matrix
        orth_blocks = []
        
        for i in range(self.num_blocks):
            # Skew-symmetric matrix
            skew = self.oft_blocks[i] - self.oft_blocks[i].transpose(-1, -2)
            # Orthogonal matrix via matrix exponential
            orth_block = torch.matrix_exp(skew)
            orth_blocks.append(orth_block)
        
        # Create block diagonal matrix
        return torch.block_diag(*orth_blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output  
        base_output = self.base_layer(x)
        
        # Apply orthogonal transformation
        orth_matrix = self.get_orthogonal_matrix()
        transformed_output = base_output @ orth_matrix.T
        
        if self.dropout:
            transformed_output = self.dropout(transformed_output)
        
        return transformed_output


class PiSSALinear(nn.Module):
    """PiSSA (Principal Singular values and Singular vectors Adaptation) implementation"""
    
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: int = 16, 
                 init_method: str = "svd", niter: int = 16, dropout: float = 0.1):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Initialize with SVD decomposition
        if init_method == "svd":
            self._init_with_svd(niter)
        else:
            self._init_random()
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def _init_with_svd(self, niter: int):
        """Initialize using SVD decomposition"""
        with torch.no_grad():
            # Perform SVD on base weight matrix
            weight = self.base_layer.weight.data
            U, S, V = torch.svd_lowrank(weight, q=self.rank, niter=niter)
            
            # Create residual weight and adaptation matrices
            self.base_layer.weight.data = weight - U @ torch.diag(S) @ V.T
            
            # Adaptation matrices initialized with principal components
            self.lora_A = nn.Linear(self.base_layer.in_features, self.rank, bias=False)
            self.lora_B = nn.Linear(self.rank, self.base_layer.out_features, bias=False)
            
            self.lora_A.weight.data = V.T
            self.lora_B.weight.data = U @ torch.diag(S)
    
    def _init_random(self):
        """Random initialization fallback"""
        self.lora_A = nn.Linear(self.base_layer.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, self.base_layer.out_features, bias=False)
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output
        base_output = self.base_layer(x)
        
        # Adaptation
        adaptation = self.lora_B(self.lora_A(x))
        if self.dropout:
            adaptation = self.dropout(adaptation)
        
        return base_output + self.scaling * adaptation


def create_peft_layer(base_layer: nn.Module, config: PEFTConfig) -> nn.Module:
    """Factory function to create PEFT layers"""
    
    if not isinstance(base_layer, nn.Linear):
        return base_layer  # Only support Linear layers for now
    
    if config.method == PEFTMethod.DORA:
        return DoRALinear(
            base_layer=base_layer,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout
        )
    
    elif config.method == PEFTMethod.OFT:
        return OFTLinear(
            base_layer=base_layer,
            block_size=config.oft_block_size,
            dropout=config.oft_dropout
        )
    
    elif config.method == PEFTMethod.PISSA:
        return PiSSALinear(
            base_layer=base_layer,
            rank=config.rank,
            alpha=config.alpha,
            init_method=config.pissa_init_method,
            niter=config.pissa_iter,
            dropout=config.dropout
        )
    
    else:
        # Fallback to original layer for unsupported methods
        return base_layer


def apply_peft_to_model(model: nn.Module, config: PEFTConfig) -> nn.Module:
    """Apply PEFT method to model"""
    
    target_modules = config.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    def replace_layers(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this layer should be replaced
            if any(target in full_name for target in target_modules):
                if isinstance(child, nn.Linear):
                    # Replace with PEFT layer
                    peft_layer = create_peft_layer(child, config)
                    setattr(module, name, peft_layer)
                    print(f"Replaced {full_name} with {config.method.value}")
            else:
                # Recursively process child modules
                replace_layers(child, full_name)
    
    replace_layers(model)
    return model


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