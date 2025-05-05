"""
Finetuner Agent Module
Specialized agent for handling model fine-tuning workflows using PEFT techniques.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from backend.app.agents.agent_core import LLMAgent
from backend.app.utils.llm import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)

class FinetunerAgent(LLMAgent):
    """Agent specialized in PEFT-style model fine-tuning workflows."""

    def __init__(
        self,
        model: str = "llama3-70b-8192",
        provider: str = "groq",
        agent_id: Optional[str] = None,
        base_model: str = "meta-llama/Llama-2-7b-hf",
        **kwargs
    ):
        """
        Initialize the finetuning agent.
        
        Args:
            model: LLM model for agent communication
            provider: LLM provider service
            agent_id: Optional unique ID for this agent
            base_model: Base model to fine-tune
        """
        system_prompt = """
        As a Fine-tuning Agent, I specialize in:
        1. Preparing datasets for fine-tuning
        2. Configuring PEFT parameters (LoRA, QLoRA, etc.)
        3. Managing training workflows
        4. Evaluating fine-tuned models
        5. Providing fine-tuning recommendations
        
        I'll help optimize models while maintaining efficiency and performance.
        """
        
        super().__init__(
            model=model,
            provider=provider,
            agent_id=agent_id or "finetuner",
            system_prompt=system_prompt,
            **kwargs
        )
        
        self.base_model = base_model
        self.current_training = None

    async def prepare_for_finetuning(
        self,
        dataset: List[Dict[str, str]],
        task_type: str = "CAUSAL_LM",
        quantization: str = "4bit"
    ) -> Dict[str, Any]:
        """
        Prepare a model and dataset for fine-tuning.
        
        Args:
            dataset: List of training examples
            task_type: Type of task (CAUSAL_LM, SEQ_CLS, etc.)
            quantization: Quantization level
            
        Returns:
            Prepared model and configuration
        """
        try:
            # Load base model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                load_in_4bit=quantization == "4bit",
                load_in_8bit=quantization == "8bit",
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            
            # Prepare for k-bit training if using quantization
            if quantization in ["4bit", "8bit"]:
                model = prepare_model_for_kbit_training(model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=16,  # Rank
                lora_alpha=32,
                target_modules=["query_key_value"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType[task_type]
            )
            
            # Get PEFT model
            peft_model = get_peft_model(model, lora_config)
            
            # Prepare training configuration
            training_args = TrainingArguments(
                output_dir="./finetuned_model",
                learning_rate=2e-4,
                num_train_epochs=3,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                save_steps=100,
                logging_steps=100,
                eval_steps=100
            )
            
            self.current_training = {
                "model": peft_model,
                "tokenizer": tokenizer,
                "config": lora_config,
                "training_args": training_args
            }
            
            return {
                "status": "success",
                "model_size": sum(p.numel() for p in peft_model.parameters()),
                "trainable_params": sum(p.numel() for p in peft_model.parameters() if p.requires_grad),
                "config": lora_config.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error preparing for fine-tuning: {str(e)}")
            raise

    async def run_finetuning(
        self,
        train_dataset: List[Dict[str, str]],
        eval_dataset: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Run the fine-tuning process.
        
        Args:
            train_dataset: Training examples
            eval_dataset: Optional evaluation examples
            
        Returns:
            Training results
        """
        if not self.current_training:
            raise ValueError("Model not prepared for fine-tuning. Call prepare_for_finetuning first.")
        
        try:
            # Convert datasets to proper format
            # ... (dataset preparation code)
            
            # Run training
            trainer = Trainer(
                model=self.current_training["model"],
                args=self.current_training["training_args"],
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
            )
            
            # Train and evaluate
            train_result = trainer.train()
            metrics = train_result.metrics
            
            if eval_dataset:
                eval_result = trainer.evaluate()
                metrics.update(eval_result)
            
            return {
                "status": "success",
                "metrics": metrics,
                "output_dir": self.current_training["training_args"].output_dir
            }
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            raise

    async def analyze_results(self, metrics: Dict[str, float]) -> str:
        """
        Analyze fine-tuning results and provide recommendations.
        
        Args:
            metrics: Training/evaluation metrics
            
        Returns:
            Analysis and recommendations
        """
        analysis_prompt = f"""
        Analyze the following fine-tuning metrics and provide recommendations:
        {json.dumps(metrics, indent=2)}
        
        Focus on:
        1. Training stability and convergence
        2. Overfitting/underfitting indicators
        3. Learning rate and batch size appropriateness
        4. Suggestions for improvement
        """
        
        config = LLMConfig(
            model=self.model,
            provider=LLMProvider(self.provider),
            temperature=0.3
        )
        
        analysis, _ = await self.llm.generate(prompt=analysis_prompt, config=config)
        return analysis

    async def run(self, user_input: str) -> str:
        """
        Process user requests related to fine-tuning.
        
        Args:
            user_input: The user's request
            
        Returns:
            Response based on the request
        """
        try:
            # Add user message to history
            self.add_message("user", user_input)
            
            # Generate appropriate response based on request type
            if "prepare" in user_input.lower():
                # Handle preparation request
                config = LLMConfig(
                    model=self.model,
                    provider=LLMProvider(self.provider),
                    temperature=0.3
                )
                response, _ = await self.llm.generate(
                    prompt=f"For the fine-tuning request: {user_input}\n\nProvide recommended PEFT configuration and preparation steps.",
                    config=config
                )
            elif "train" in user_input.lower() or "finetune" in user_input.lower():
                # Handle training request
                config = LLMConfig(
                    model=self.model,
                    provider=LLMProvider(self.provider),
                    temperature=0.3
                )
                response, _ = await self.llm.generate(
                    prompt=f"For the fine-tuning request: {user_input}\n\nProvide training configuration and monitoring recommendations.",
                    config=config
                )
            else:
                # General fine-tuning guidance
                config = LLMConfig(
                    model=self.model,
                    provider=LLMProvider(self.provider),
                    temperature=0.7
                )
                response, _ = await self.llm.generate(prompt=user_input, config=config)
            
            # Add response to history
            self.add_message("assistant", response)
            return response
            
        except Exception as e:
            logger.error(f"Error in finetuner agent: {str(e)}")
            return f"I encountered an error while processing your fine-tuning request: {str(e)}"