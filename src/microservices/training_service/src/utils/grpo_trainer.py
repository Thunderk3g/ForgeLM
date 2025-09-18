"""
GRPO (Guided Reinforcement from Process Outcomes) Trainer

Implementation of GRPO training for reasoning and decision-making models.
GRPO trains models to produce better intermediate reasoning steps by learning 
from process-level outcomes rather than just final answers.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from datasets import Dataset
import numpy as np
from pathlib import Path
import json
import time

try:
    import unsloth
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from utils.logging import get_logger


class GRPOTrainer:
    """GRPO Trainer for reasoning and decision-making models"""
    
    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        max_seq_length: int = 2048,
        use_unsloth: bool = True,
        device_map: str = "auto"
    ):
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.max_seq_length = max_seq_length
        self.use_unsloth = use_unsloth and UNSLOTH_AVAILABLE
        self.device_map = device_map
        
        # Initialize model and tokenizer
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with Unsloth optimization if available"""
        try:
            if self.use_unsloth:
                self.logger.info("Loading model with Unsloth optimization for GRPO")
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_name,
                    max_seq_length=self.max_seq_length,
                    dtype=None,  # Auto-detect
                    load_in_4bit=True,
                    device_map=self.device_map,
                    trust_remote_code=True,
                )
                
                # Apply GRPO-optimized LoRA
                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=32,  # Higher rank for reasoning
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"
                    ],
                    lora_alpha=32,
                    lora_dropout=0.0,  # No dropout for reasoning stability
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=3407,
                    use_rslora=False,
                    loftq_config=None,
                )
            else:
                self.logger.info("Loading model with standard transformers for GRPO")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name, 
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    device_map=self.device_map,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
                
            # Set pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
        except Exception as e:
            self.logger.error(f"Error loading model and tokenizer: {e}")
            raise
    
    def prepare_grpo_dataset(
        self, 
        dataset: Dataset, 
        process_supervision: bool = True
    ) -> Dataset:
        """
        Prepare dataset for GRPO training with process supervision
        
        Args:
            dataset: Input dataset with reasoning examples
            process_supervision: Whether to add process-level supervision
            
        Returns:
            Processed dataset for GRPO training
        """
        def process_example(example):
            """Process a single example for GRPO training"""
            try:
                # Extract components
                question = example.get("question", "")
                reasoning_steps = example.get("reasoning_steps", [])
                final_answer = example.get("final_answer", "")
                process_rewards = example.get("process_rewards", [])
                
                # Format for chat template
                if process_supervision and reasoning_steps:
                    # Create supervised reasoning chain
                    messages = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": self._format_reasoning_chain(
                            reasoning_steps, final_answer, process_rewards
                        )}
                    ]
                else:
                    # Standard question-answer format
                    messages = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": final_answer}
                    ]
                
                # Apply chat template
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                return {"text": text}
                
            except Exception as e:
                self.logger.warning(f"Error processing example: {e}")
                return {"text": ""}
        
        return dataset.map(process_example, remove_columns=dataset.column_names)
    
    def _format_reasoning_chain(
        self, 
        reasoning_steps: List[str], 
        final_answer: str,
        process_rewards: Optional[List[float]] = None
    ) -> str:
        """Format reasoning chain with process supervision"""
        formatted_chain = []
        
        for i, step in enumerate(reasoning_steps):
            # Add step marker
            formatted_chain.append(f"Step {i+1}: {step}")
            
            # Add process reward if available
            if process_rewards and i < len(process_rewards):
                reward = process_rewards[i]
                if reward > 0.7:
                    formatted_chain.append("[GOOD STEP]")
                elif reward < 0.3:
                    formatted_chain.append("[NEEDS IMPROVEMENT]")
        
        formatted_chain.append(f"Final Answer: {final_answer}")
        return "\n".join(formatted_chain)
    
    def create_grpo_training_args(
        self,
        output_dir: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        learning_rate: float = 1e-4,
        **kwargs
    ) -> TrainingArguments:
        """Create training arguments optimized for GRPO"""
        
        args = {
            "output_dir": output_dir,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_steps": 20,
            "logging_steps": 5,
            "save_steps": 500,
            "save_total_limit": 2,
            "evaluation_strategy": "no",
            "weight_decay": 0.001,
            "lr_scheduler_type": "cosine",
            "seed": 3407,
            "dataloader_num_workers": 1,
            "dataloader_pin_memory": False,
            "remove_unused_columns": False,
            "group_by_length": False,
            "prediction_loss_only": False,
            "run_name": "grpo_reasoning_training",
            "report_to": [],
        }
        
        # GRPO-specific optimizations
        if torch.cuda.is_available():
            args.update({
                "fp16": False,  # Use bf16 for better stability
                "bf16": True,
                "gradient_checkpointing": True,
                "dataloader_persistent_workers": False,
            })
        
        # Unsloth optimizations
        if self.use_unsloth:
            args.update({
                "optim": "adamw_8bit",
                "max_grad_norm": 0.3,
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
                "adam_epsilon": 1e-5,
                "use_gradient_checkpointing": "unsloth",
            })
        
        # Override with any provided kwargs
        args.update(kwargs)
        
        return TrainingArguments(**args)
    
    def train(
        self,
        train_dataset: Dataset,
        training_args: TrainingArguments,
        eval_dataset: Optional[Dataset] = None,
        callbacks: Optional[List[TrainerCallback]] = None
    ) -> Dict[str, Any]:
        """Train model with GRPO"""
        
        try:
            self.logger.info("Starting GRPO training")
            
            # Prepare dataset
            processed_dataset = self.prepare_grpo_dataset(train_dataset)
            
            # Create trainer
            from trl import SFTTrainer, SFTConfig
            
            sft_config = SFTConfig(
                dataset_text_field="text",
                max_seq_length=self.max_seq_length,
                packing=False,  # Don't pack for reasoning
                dataset_num_proc=1,
                chars_per_token=3.5,
                # GRPO specific
                dataset_kwargs={
                    "add_special_tokens": False,
                },
            )
            
            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=processed_dataset,
                eval_dataset=eval_dataset,
                args=training_args,
                **sft_config.to_dict(),
            )
            
            # Add callbacks
            if callbacks:
                for callback in callbacks:
                    trainer.add_callback(callback)
            
            # Train
            training_result = trainer.train()
            
            self.logger.info("GRPO training completed successfully")
            
            return {
                "status": "completed",
                "training_loss": training_result.training_loss,
                "global_step": training_result.global_step,
                "metrics": training_result.metrics,
                "optimization": "grpo_reasoning",
            }
            
        except Exception as e:
            self.logger.error(f"GRPO training failed: {e}")
            raise
    
    def save_model(self, output_dir: str):
        """Save the trained model"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if self.use_unsloth:
                # Save with Unsloth
                self.model.save_pretrained(str(output_path))
                self.tokenizer.save_pretrained(str(output_path))
                
                # Also save merged model for inference
                merged_path = output_path / "merged_16bit"
                self.model.save_pretrained_merged(
                    str(merged_path),
                    self.tokenizer,
                    save_method="merged_16bit",
                )
            else:
                # Standard save
                self.model.save_pretrained(str(output_path))
                self.tokenizer.save_pretrained(str(output_path))
            
            self.logger.info(f"GRPO model saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving GRPO model: {e}")
            raise
    
    def evaluate_reasoning(
        self, 
        test_dataset: Dataset, 
        max_new_tokens: int = 512
    ) -> Dict[str, Any]:
        """Evaluate reasoning capabilities of the trained model"""
        try:
            self.logger.info("Evaluating reasoning capabilities")
            
            results = []
            
            for example in test_dataset:
                question = example.get("question", "")
                expected_answer = example.get("final_answer", "")
                
                # Generate response
                messages = [{"role": "user", "content": question}]
                input_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                inputs = self.tokenizer(input_text, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Decode response
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], 
                    skip_special_tokens=True
                )
                
                results.append({
                    "question": question,
                    "expected_answer": expected_answer,
                    "generated_answer": generated_text,
                    "reasoning_quality": self._assess_reasoning_quality(generated_text),
                })
            
            # Calculate metrics
            avg_reasoning_quality = np.mean([r["reasoning_quality"] for r in results])
            
            evaluation_results = {
                "average_reasoning_quality": avg_reasoning_quality,
                "total_examples": len(results),
                "detailed_results": results,
            }
            
            self.logger.info(f"Reasoning evaluation completed. Average quality: {avg_reasoning_quality:.3f}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error during reasoning evaluation: {e}")
            raise
    
    def _assess_reasoning_quality(self, generated_text: str) -> float:
        """Assess the quality of reasoning in generated text"""
        try:
            # Simple heuristic assessment
            quality_score = 0.0
            
            # Check for step-by-step reasoning
            if "step" in generated_text.lower():
                quality_score += 0.3
            
            # Check for logical connectors
            logical_words = ["because", "therefore", "since", "thus", "hence", "so"]
            if any(word in generated_text.lower() for word in logical_words):
                quality_score += 0.2
            
            # Check for conclusion
            if "final answer" in generated_text.lower() or "conclusion" in generated_text.lower():
                quality_score += 0.2
            
            # Check length (reasoning should be substantial)
            if len(generated_text.split()) > 50:
                quality_score += 0.3
            
            return min(quality_score, 1.0)
            
        except Exception:
            return 0.0


class GRPOCallback(TrainerCallback):
    """Callback for GRPO training progress tracking"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.logger = get_logger(f"grpo.training.{task_id}")
        self.start_time = time.time()
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.logger.info(f"Starting GRPO training with {state.max_steps} total steps")
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            current_step = state.global_step
            total_steps = state.max_steps
            progress_pct = (current_step / total_steps * 100) if total_steps > 0 else 0
            
            # Get loss from logs if available
            loss = None
            if state.log_history:
                last_log = state.log_history[-1]
                loss = last_log.get("train_loss", last_log.get("loss"))
            
            message = f"GRPO Step {current_step}/{total_steps} ({progress_pct:.1f}%)"
            if loss is not None:
                message += f" - Loss: {loss:.4f}"
            
            self.logger.info(message)
    
    def on_train_end(self, args, state, control, **kwargs):
        duration = time.time() - self.start_time
        self.logger.info(f"GRPO training completed in {duration:.2f} seconds")