import pandas as pd
import random
from transformers import pipeline, T5TokenizerFast, T5ForConditionalGeneration
import torch
from typing import List, Dict
import re
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from functools import partial
import os

# Suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

class ModelBasedRAGAugmentor:
    def __init__(self, use_multi_gpu=True):
        self.use_multi_gpu = use_multi_gpu
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if self.use_multi_gpu and self.num_gpus > 1:
            print(f"Multi-GPU mode enabled: Using {self.num_gpus} GPUs")
            self.paraphrase_models = []
            self.question_generators = []
            self.load_models_multi_gpu()
        else:
            print(f"Single GPU/CPU mode: Using {'GPU' if self.num_gpus > 0 else 'CPU'}")
            self.paraphrase_model = None
            self.question_generator = None
            self.load_models_single()
    
    def load_models_multi_gpu(self):
        """Load models across multiple GPUs"""
        print("Loading models across multiple GPUs...")
        
        # Try better paraphrasing models in order of preference
        model_options = [
            "Vamsi/T5_Paraphrase_Paws",  # Better T5 paraphraser
            "tuner007/pegasus_paraphrase", # PEGASUS-based
            "prithivida/parrot_paraphraser_on_T5", # Parrot paraphraser
            "ramsrigouthamg/t5_paraphraser"  # Fallback
        ]
        
        for gpu_id in range(self.num_gpus):
            print(f"Loading paraphrasing model on GPU {gpu_id}...")
            model_loaded = False
            
            for model_name in model_options:
                try:
                    print(f"  Trying {model_name}...")
                    paraphrase_model = pipeline(
                        "text2text-generation",
                        model=model_name,
                        device=gpu_id,
                        max_length=128,
                        do_sample=True,
                        temperature=0.8,
                        num_return_sequences=1
                    )
                    self.paraphrase_models.append(paraphrase_model)
                    print(f"  ✓ Successfully loaded {model_name} on GPU {gpu_id}")
                    model_loaded = True
                    break
                    
                except Exception as e:
                    print(f"  ✗ Failed to load {model_name}: {e}")
                    continue
            
            if not model_loaded:
                print(f"  Using fallback: copying GPU 0 model")
                if gpu_id > 0 and self.paraphrase_models:
                    self.paraphrase_models.append(self.paraphrase_models[0])
    
    def load_models_single(self):
        """Load models on single GPU or CPU"""
        device = 0 if self.num_gpus > 0 else -1
        
        # Try better paraphrasing models
        model_options = [
            ("Vamsi/T5_Paraphrase_Paws", "paraphrase: "),
            ("tuner007/pegasus_paraphrase", "paraphrase: "),
            ("prithivida/parrot_paraphraser_on_T5", "paraphrase: "),
            ("ramsrigouthamg/t5_paraphraser", "paraphrase: ")
        ]
        
        for model_name, prompt_format in model_options:
            try:
                print(f"Trying to load {model_name}...")
                self.paraphrase_model = pipeline(
                    "text2text-generation",
                    model=model_name,
                    device=device,
                    max_length=128,
                    do_sample=True,
                    temperature=0.8,
                    num_return_sequences=1
                )
                self.prompt_format = prompt_format
                print(f"✓ Successfully loaded {model_name}")
                break
                
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
                continue
        
        if not hasattr(self, 'paraphrase_model') or self.paraphrase_model is None:
            print("Could not load any paraphrasing model. Will use rule-based generation only.")
            self.prompt_format = "paraphrase: "
    
    def get_model_for_gpu(self, gpu_id):
        """Get models for specific GPU"""
        if self.use_multi_gpu and self.paraphrase_models:
            model_idx = gpu_id % len(self.paraphrase_models)
            return self.paraphrase_models[model_idx]
        else:
            return self.paraphrase_model

    def detect_dataset_type(self, sample_questions: List[str]) -> str:
        """Auto-detect dataset type based on question patterns"""
        evqa_indicators = 0
        infoseek_indicators = 0
        
        for question in sample_questions[:50]:
            question_lower = question.lower()
            
            # EVQA indicators
            if any(indicator in question_lower for indicator in [
                'do the', 'does this', 'is this', 'are the', 'how many', 'what stage',
                'according to', 'in what language', 'who first described',
                'religion', 'folklore', 'native americans', 'subspecies'
            ]):
                evqa_indicators += 1
                
            # InfoSeek indicators  
            if any(indicator in question_lower for indicator in [
                'what product', 'who is the manufacturer', 'what is the closest',
                'taxonomy', 'what country does', 'what fields are',
                'what culture is associated', 'who operates'
            ]):
                infoseek_indicators += 1
        
        return 'evqa' if evqa_indicators > infoseek_indicators else 'infoseek'

    def paraphrase_question_multi_gpu(self, question: str, gpu_id: int = 0) -> List[str]:
        """Generate paraphrases using specific GPU"""
        variations = []
        
        paraphrase_model, _ = self.get_model_for_gpu(gpu_id)
        
        if paraphrase_model:
            try:
                input_text = f"paraphrase: {question}"
                result = paraphrase_model(input_text, max_length=128, num_return_sequences=3)
                
                for item in result:
                    paraphrased = item['generated_text'].strip()
                    if paraphrased and paraphrased != question and paraphrased not in variations:
                        variations.append(paraphrased)
                        
            except Exception as e:
                print(f"Paraphrase error on GPU {gpu_id}: {e}")
        
        # Fallback to rule-based variations
        rule_based = self.rule_based_variations(question)
        if rule_based:
            variations.extend(rule_based)
        
        return variations if variations else []

    def process_batch_multi_gpu(self, batch_data):
        """Process a batch of questions on specific GPU"""
        batch_questions, dataset_type, variations_per_question, gpu_id = batch_data
        
        # Set the GPU for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        
        results = []
        
        for question_data in batch_questions:
            original_question = question_data['question']
            
            # Generate variations
            variations = self.augment_single_question_gpu(
                original_question, 
                dataset_type, 
                variations_per_question,
                gpu_id
            )
            
            # Create new rows for each variation
            for variation in variations:
                new_row = question_data.copy()
                new_row['question'] = variation
                new_row['dataset'] = 'augmented'
                results.append(new_row)
        
        return results

    def augment_single_question_gpu(self, question: str, dataset_type: str, target_variations: int = 10, gpu_id: int = 0) -> List[str]:
        """Generate variations using specific GPU"""
        all_variations = []
        
        # Method 1: GPU-based paraphrasing
        paraphrases = self.paraphrase_question_multi_gpu(question, gpu_id)
        all_variations.extend(paraphrases)
        
        # Method 2: Rule-based variations (CPU)
        linguistic_vars = self.entity_substitution(question, dataset_type)
        all_variations.extend(linguistic_vars)
        
        advanced_vars = self.advanced_linguistic_variations(question)
        all_variations.extend(advanced_vars)
        
        # Remove duplicates and filter
        unique_variations = []
        seen = set([question.lower()])
        
        for var in all_variations:
            var_clean = var.strip()
            if (var_clean and 
                len(var_clean) > 10 and 
                var_clean.lower() not in seen and
                var_clean.endswith('?')):
                unique_variations.append(var_clean)
                seen.add(var_clean.lower())
                
                if len(unique_variations) >= target_variations:
                    break
        
        return unique_variations

    def print_examples(self, original_questions: List[str], dataset_type: str, num_examples: int = 3):
        """Print examples of generated variations"""
        print(f"\n=== EXAMPLE VARIATIONS (Dataset: {dataset_type.upper()}) ===")
        
        for i, original_q in enumerate(original_questions[:num_examples]):
            print(f"\n--- Example {i+1} ---")
            print(f"Original: {original_q}")
            
            variations = self.augment_single_question(original_q, dataset_type, target_variations=8)
            
            if variations:
                print("Generated variations:")
                for j, var in enumerate(variations[:6], 1):  # Show max 6 variations
                    print(f"  {j}. {var}")
            else:
                print("  No variations generated")
        
        print(f"\n{'='*60}")
        response = input("Continue with full data augmentation? (y/n): ")
        return response.lower().startswith('y')

    def paraphrase_question(self, question: str) -> List[str]:
        """Generate paraphrases of a question using T5 model"""
        variations = []
        
        # Check for paraphrase model in both single and multi-GPU modes
        paraphrase_model = None
        if hasattr(self, 'paraphrase_model') and self.paraphrase_model:
            paraphrase_model = self.paraphrase_model
        elif hasattr(self, 'paraphrase_models') and self.paraphrase_models:
            paraphrase_model = self.paraphrase_models[0]  # Use first GPU's model for examples
        
        if paraphrase_model:
            try:
                # T5 paraphraser expects specific format
                input_text = f"paraphrase: {question}"
                result = paraphrase_model(input_text, max_length=128, num_return_sequences=3)
                
                for item in result:
                    paraphrased = item['generated_text'].strip()
                    if paraphrased and paraphrased != question and paraphrased not in variations:
                        variations.append(paraphrased)
                        
            except Exception as e:
                print(f"Paraphrase error: {e}")
        
        # Fallback to rule-based variations
        rule_based = self.rule_based_variations(question)
        if rule_based:
            variations.extend(rule_based)
        
        return variations if variations else []

    def augment_single_question(self, question: str, dataset_type: str, target_variations: int = 10) -> List[str]:
        """Generate multiple variations - using better models"""
        all_variations = []
        
        # Get the correct model
        paraphrase_model = None
        prompt_format = "paraphrase: "  # default
        
        if hasattr(self, 'paraphrase_model') and self.paraphrase_model:
            paraphrase_model = self.paraphrase_model
            prompt_format = getattr(self, 'prompt_format', "paraphrase: ")
        elif hasattr(self, 'paraphrase_models') and self.paraphrase_models:
            paraphrase_model = self.paraphrase_models[0]
            prompt_format = "paraphrase: "
        
        # METHOD 1: Better T5/PEGASUS paraphrasing with multiple approaches
        if paraphrase_model:
            print(f"DEBUG: Using paraphrasing model")
            
            # Try different prompting approaches
            prompts = [
                f"paraphrase: {question}",
                f"rephrase: {question}",
                f"rewrite: {question}",
                question  # Some models don't need prefix
            ]
            
            # Try different parameters
            configs = [
                {"temperature": 0.7, "top_p": 0.9, "num_beams": 3},
                {"temperature": 0.8, "top_p": 0.85, "do_sample": True},
                {"temperature": 0.9, "top_p": 0.9, "do_sample": True},
                {"temperature": 1.0, "top_p": 0.95, "do_sample": True},
                {"temperature": 1.1, "top_p": 0.9, "do_sample": True}
            ]
            
            for prompt in prompts:
                for config in configs:
                    try:
                        result = paraphrase_model(
                            prompt,
                            max_length=128,
                            num_return_sequences=2,
                            **config
                        )
                        
                        for item in result:
                            paraphrased = item['generated_text'].strip()
                            
                            # Clean up the output (remove prompt if it's echoed)
                            for prefix in ["paraphrase:", "rephrase:", "rewrite:"]:
                                if paraphrased.lower().startswith(prefix):
                                    paraphrased = paraphrased[len(prefix):].strip()
                            
                            if (paraphrased and 
                                paraphrased != question and 
                                paraphrased.endswith('?') and
                                paraphrased not in all_variations and
                                len(paraphrased) > 10):
                                all_variations.append(paraphrased)
                                
                        # Break early if we have enough variations
                        if len(all_variations) >= target_variations * 2:
                            break
                            
                    except Exception as e:
                        continue
                        
                if len(all_variations) >= target_variations * 2:
                    break
        
        print(f"DEBUG: Model generated {len(all_variations)} variations")
        
        # METHOD 2: Rule-based as backup (only if model fails completely)
        if len(all_variations) < target_variations // 3:
            print("DEBUG: Model insufficient, adding rule-based backup")
            rule_vars = self.rule_based_variations(question)
            all_variations.extend(rule_vars)
            
            entity_vars = self.entity_substitution(question, dataset_type)
            all_variations.extend(entity_vars)
        
        # Filter and deduplicate
        unique_variations = []
        seen = set([question.lower()])
        
        for var in all_variations:
            var_clean = var.strip()
            if (var_clean and 
                len(var_clean) > 10 and 
                var_clean.lower() not in seen and
                var_clean.endswith('?')):
                unique_variations.append(var_clean)
                seen.add(var_clean.lower())
                
                if len(unique_variations) >= target_variations:
                    break
        
        print(f"DEBUG: Final unique variations: {len(unique_variations)}")
        return unique_variations

    def generate_multiple_paraphrases(self, question: str, num_attempts: int = 3) -> List[str]:
        """Generate multiple rounds of paraphrases with different settings"""
        all_paraphrases = []
        
        paraphrase_model = None
        if hasattr(self, 'paraphrase_model') and self.paraphrase_model:
            paraphrase_model = self.paraphrase_model
        elif hasattr(self, 'paraphrase_models') and self.paraphrase_models:
            paraphrase_model = self.paraphrase_models[0]
        
        if paraphrase_model:
            for attempt in range(num_attempts):
                try:
                    # Vary parameters for diversity
                    temperature = 0.7 + (attempt * 0.1)  # 0.7, 0.8, 0.9
                    
                    input_text = f"paraphrase: {question}"
                    result = paraphrase_model(
                        input_text, 
                        max_length=128, 
                        num_return_sequences=4,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.9
                    )
                    
                    for item in result:
                        paraphrased = item['generated_text'].strip()
                        if (paraphrased and 
                            paraphrased != question and 
                            paraphrased not in all_paraphrases and
                            paraphrased.endswith('?')):
                            all_paraphrases.append(paraphrased)
                            
                except Exception as e:
                    print(f"Paraphrase attempt {attempt + 1} failed: {e}")
                    
        return all_paraphrases

    def generate_contextual_questions(self, question: str) -> List[str]:
        """Generate questions using contextual understanding"""
        variations = []
        
        question_gen = None
        if hasattr(self, 'question_generator') and self.question_generator:
            question_gen = self.question_generator
        elif hasattr(self, 'question_generators') and self.question_generators:
            question_gen = self.question_generators[0]
        
        if question_gen:
            # Multiple context approaches
            contexts = [
                "This image shows a plant that can be identified.",
                "The biological specimen in the image has specific characteristics.",
                "This visual content depicts a natural element with geographic properties."
            ]
            
            for context in contexts:
                try:
                    input_text = f"context: {context} question: {question}"
                    result = question_gen(
                        input_text, 
                        max_length=64, 
                        num_return_sequences=3,
                        temperature=0.8,
                        do_sample=True
                    )
                    
                    for item in result:
                        generated = item['generated_text'].strip()
                        if (generated and 
                            generated.endswith('?') and
                            generated not in variations and
                            len(generated) > 10):
                            variations.append(generated)
                            
                except Exception as e:
                    print(f"Context generation error: {e}")
                    
        return variations

    def generate_semantic_variations(self, question: str) -> List[str]:
        """Generate variations by modifying semantic components"""
        variations = []
        
        paraphrase_model = None
        if hasattr(self, 'paraphrase_model') and self.paraphrase_model:
            paraphrase_model = self.paraphrase_model
        elif hasattr(self, 'paraphrase_models') and self.paraphrase_models:
            paraphrase_model = self.paraphrase_models[0]
            
        if paraphrase_model:
            # Semantic modification prompts
            semantic_prompts = [
                f"rephrase this question more formally: {question}",
                f"ask the same thing differently: {question}",
                f"reformulate: {question}",
                f"rewrite this question: {question}"
            ]
            
            for prompt in semantic_prompts:
                try:
                    result = paraphrase_model(
                        prompt,
                        max_length=100,
                        num_return_sequences=2,
                        temperature=0.8
                    )
                    
                    for item in result:
                        semantic_var = item['generated_text'].strip()
                        if (semantic_var and 
                            semantic_var.endswith('?') and
                            semantic_var not in variations and
                            semantic_var != question):
                            variations.append(semantic_var)
                            
                except Exception as e:
                    print(f"Semantic generation error: {e}")
                    
        return variations
        """Generate paraphrases of a question using T5 model"""
        variations = []
        
        if self.paraphrase_model:
            try:
                # T5 paraphraser expects specific format
                input_text = f"paraphrase: {question}"
                result = self.paraphrase_model(input_text, max_length=128, num_return_sequences=3)
                
                for item in result:
                    paraphrased = item['generated_text'].strip()
                    if paraphrased and paraphrased != question and paraphrased not in variations:
                        variations.append(paraphrased)
                        
            except Exception as e:
                print(f"Paraphrase error: {e}")
        
        # Fallback to rule-based variations
        rule_based = self.rule_based_variations(question)
        if rule_based:  # Check if not None or empty
            variations.extend(rule_based)
        
        return variations if variations else []  # Always return a list

    def rule_based_variations(self, question: str) -> List[str]:
        """Generate variations using linguistic rules"""
        variations = []
        
        # Question word substitutions
        substitutions = {
            'What': ['Which', 'What type of', 'What kind of'],
            'Which': ['What', 'What type of', 'What sort of'],
            'Who': ['What person', 'Which individual'],
            'Where': ['In which location', 'At what place'],
            'How': ['In what way', 'By what means'],
            'When': ['At what time', 'During which period']
        }
        
        for original, replacements in substitutions.items():
            if question.startswith(original):
                for replacement in replacements:
                    new_question = question.replace(original, replacement, 1)
                    variations.append(new_question)
        
        # Structural variations
        if question.startswith('What') and 'does this' in question:
            # "What does this X do?" -> "This X does what?"
            match = re.search(r'What does this (.+?) (.+)\?', question)
            if match:
                entity, action = match.groups()
                variations.append(f"This {entity} {action} what?")
        
        if question.startswith('Is this'):
            # "Is this X Y?" -> "This X is Y or not?"
            variations.append(question.replace('Is this', 'This').replace('?', ' or not?'))
        
    def advanced_linguistic_variations(self, question: str) -> List[str]:
        """Generate variations using advanced linguistic transformations"""
        variations = []
        
        # More sophisticated sentence structure transformations
        transformations = [
            # Direct to indirect with variety
            (r"In which (.+?) does (.+?)\?", [
                r"Can you identify the \1 where \2?",
                r"What \1 is \2 found in?",
                r"Which \1 serves as the location for \2?",
                r"\2 is located in which \1?"
            ]),
            
            # What/Which variations with context
            (r"What (.+?) does (.+?)\?", [
                r"Can you tell me what \1 \2?",
                r"Which \1 does \2?",
                r"What kind of \1 does \2?",
                r"\2 has what \1?"
            ]),
            
            # Location-specific improvements
            (r"(.+?) grow\?", [
                r"\1 thrive?",
                r"\1 flourish?",
                r"\1 naturally occur?",
                r"\1 have their habitat?"
            ]),
            
            # Add contextual depth
            (r"this (.+?) (.+?)\?", [
                r"the \1 in the image \2?",
                r"the \1 shown \2?",
                r"the depicted \1 \2?",
                r"the \1 we see \2?"
            ])
        ]
        
        for pattern, replacements in transformations:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                for replacement in replacements[:2]:  # Limit to prevent explosion
                    try:
                        new_question = re.sub(pattern, replacement, question, flags=re.IGNORECASE)
                        if new_question != question and new_question.endswith('?'):
                            variations.append(new_question)
                    except:
                        continue
        
        return variations

    def entity_substitution(self, question: str, dataset_type: str) -> List[str]:
        """Replace entities with generic alternatives (entity-agnostic approach)"""
        variations = []
        
        try:
            # Generic substitutions that work for any entity
            generic_substitutions = [
                # Demonstratives
                ('this', ['the', 'that']),
                
                # Question starters - more natural variations
                ('What is', ['What would be', 'What might be']),
                ('What does', ['What can', 'What would']),
                ('Which', ['What', 'What type of']),
                ('Who is', ['Who was', 'Who would be']),
                ('How is', ['How was', 'How can']),
                ('Where is', ['Where was', 'Where can']),
                
                # Verbs - tense variations
                ('belong to', ['come from', 'originate from']),
                ('found in', ['located in', 'present in']),
                ('used for', ['designed for', 'intended for']),
                ('made of', ['constructed from', 'built with']),
                ('known as', ['called', 'referred to as']),
                
                # Descriptive terms
                ('type of', ['kind of', 'sort of']),
                ('made by', ['created by', 'produced by']),
                ('located in', ['situated in', 'positioned in']),
            ]
            
            for original, replacements in generic_substitutions:
                if original.lower() in question.lower():
                    for replacement in replacements[:1]:  # Use 1 replacement to avoid explosion
                        new_question = re.sub(
                            re.escape(original), 
                            replacement, 
                            question, 
                            flags=re.IGNORECASE,
                            count=1
                        )
                        if new_question != question:
                            variations.append(new_question)
        except Exception as e:
            print(f"Entity substitution error: {e}")
        
        return variations

    def generate_from_context(self, question: str) -> List[str]:
        """Placeholder - not actually used anymore"""
        return []

    def augment_single_question(self, question: str, dataset_type: str, target_variations: int = 10) -> List[str]:
        """Generate multiple variations of a single question"""
        all_variations = []
        
        # Method 1: Paraphrasing
        paraphrases = self.paraphrase_question(question)
        all_variations.extend(paraphrases)
        
        # Method 2: Generic linguistic substitution (entity-agnostic)
        linguistic_vars = self.entity_substitution(question, dataset_type)
        all_variations.extend(linguistic_vars)
        
        # Method 3: Advanced linguistic transformations
        advanced_vars = self.advanced_linguistic_variations(question)
        all_variations.extend(advanced_vars)
        
        # Method 4: Apply transformations to paraphrases
        for paraphrase in paraphrases[:2]:
            linguistic_vars_of_paraphrases = self.entity_substitution(paraphrase, dataset_type)
            all_variations.extend(linguistic_vars_of_paraphrases)
        
        # Method 5: Context-based generation (reduced weight)
        context_questions = self.generate_from_context(question)
        all_variations.extend(context_questions[:2])  # Limit to prevent low-quality generation
        
        # Remove duplicates and filter quality
        unique_variations = []
        seen = set([question.lower()])
        
        for var in all_variations:
            var_clean = var.strip()
            if (var_clean and 
                len(var_clean) > 10 and 
                var_clean.lower() not in seen and
                var_clean.endswith('?')):
                unique_variations.append(var_clean)
                seen.add(var_clean.lower())
                
                if len(unique_variations) >= target_variations:
                    break
        
        return unique_variations

    def augment_data_multi_gpu(self, original_csv_path: str, target_count: int = 200000, 
                              variations_per_question: int = 10) -> pd.DataFrame:
        """Execute multi-GPU data augmentation"""
        print(f"Target: Generate {target_count:,} rag_label=YES data using {self.num_gpus} GPUs")
        
        # Load original data
        df = pd.read_csv(original_csv_path)
        original_yes_data = df[df['rag_label'] == 'YES'].copy()
        original_count = len(original_yes_data)
        
        print(f"Original YES data: {original_count:,}")
        
        # Auto-detect dataset type
        sample_questions = original_yes_data['question'].head(50).tolist()
        dataset_type = self.detect_dataset_type(sample_questions)
        print(f"Detected dataset type: {dataset_type.upper()}")
        
        # DEBUG: Check for question duplicates in original data
        question_counts = original_yes_data['question'].value_counts()
        duplicates = question_counts[question_counts > 1]
        
        print(f"\nDEBUG: Data analysis")
        print(f"- Total YES data: {len(original_yes_data)}")
        print(f"- Unique questions: {len(question_counts)}")
        print(f"- Questions with duplicates: {len(duplicates)}")
        
        if len(duplicates) > 0:
            print(f"- Most common question appears {duplicates.iloc[0]} times")
            print(f"- Top duplicate: '{duplicates.index[0]}'")
        
        # Calculate needed augmentation
        needed_count = target_count - original_count
        print(f"Data to generate: {needed_count:,}")
        
        if needed_count <= 0:
            print("Target count already reached!")
            return original_yes_data
        
        # Show examples
        original_rows = original_yes_data.to_dict('records')
        sample_questions = [row['question'] for row in original_rows[:5]]
        should_continue = self.print_examples(sample_questions, dataset_type, num_examples=3)
        
        if not should_continue:
            print("Data augmentation cancelled by user.")
            return original_yes_data
        
        # Calculate questions needed per GPU (non-overlapping by original_index)
        total_questions = len(original_rows)
        questions_per_batch = total_questions // self.num_gpus
        remainder = total_questions % self.num_gpus
        
        print(f"Distributing {total_questions} unique questions across {self.num_gpus} GPUs")
        print(f"Base questions per GPU: {questions_per_batch}")
        if remainder > 0:
            print(f"First {remainder} GPUs will get 1 extra question each")
        
        # Create non-overlapping batches for each GPU using original_index
        batches = []
        current_start = 0
        
        for gpu_id in range(self.num_gpus):
            # First 'remainder' GPUs get one extra question
            batch_size = questions_per_batch + (1 if gpu_id < remainder else 0)
            
            if current_start < total_questions:
                end_idx = min(current_start + batch_size, total_questions)
                batch_questions = original_rows[current_start:end_idx]
                
                # Extract original indices for this batch
                batch_indices = [row['original_index'] for row in batch_questions]
                print(f"GPU {gpu_id}: indices {min(batch_indices)} to {max(batch_indices)} ({len(batch_questions)} questions)")
                
                batches.append((batch_questions, dataset_type, variations_per_question, gpu_id))
                current_start = end_idx
            else:
                break
        
        print(f"Created {len(batches)} batches for processing")
        
        # Process batches in parallel using multiprocessing
        print("Starting multi-GPU processing...")
        
        # Use spawn method for CUDA compatibility
        ctx = mp.get_context('spawn')
        
        with ctx.Pool(processes=self.num_gpus) as pool:
            batch_results = []
            
            # Process batches in parallel
            for result in tqdm(pool.imap(self.process_batch_wrapper, batches), 
                             total=len(batches), desc="Processing GPU batches"):
                batch_results.extend(result)
                
                # Stop if we have enough data
                if len(batch_results) >= needed_count:
                    break
        
        # Trim to exact needed count
        batch_results = batch_results[:needed_count]
        
        # Update data_id for results
        for i, row in enumerate(batch_results):
            row['data_id'] = f'aug_{i:06d}'
            # Remove the temporary original_index column
            if 'original_index' in row:
                del row['original_index']
        
        print(f"Generated {len(batch_results):,} new questions")
        
        # Remove original_index from original data too
        original_yes_data_clean = original_yes_data.copy()
        if 'original_index' in original_yes_data_clean.columns:
            original_yes_data_clean = original_yes_data_clean.drop('original_index', axis=1)
        
        # Create final dataframe
        augmented_df = pd.DataFrame(batch_results)
        final_df = pd.concat([original_yes_data_clean, augmented_df], ignore_index=True)
        
        print(f"\nFinal results:")
        print(f"- Original YES data: {original_count:,}")
        print(f"- Augmented data: {len(augmented_df):,}")
        print(f"- Total YES data: {len(final_df):,}")
        print(f"- Total columns maintained: {len(final_df.columns)}")
        
        return final_df

    @staticmethod
    def process_batch_wrapper(batch_data):
        """Wrapper for multiprocessing compatibility"""
        batch_questions, dataset_type, variations_per_question, gpu_id = batch_data
        
        # Create new augmentor instance for this process
        augmentor = ModelBasedRAGAugmentor(use_multi_gpu=False)  # Single GPU per process
        
        results = []
        for question_data in batch_questions:
            original_question = question_data['question']
            
            # Generate variations
            variations = augmentor.augment_single_question(
                original_question, 
                dataset_type, 
                variations_per_question
            )
            
            # Create new rows for each variation
            for variation in variations:
                new_row = question_data.copy()
                new_row['question'] = variation
                new_row['dataset'] = 'augmented'
                results.append(new_row)
        
        return results

    def augment_data(self, original_csv_path: str, target_count: int = 200000, 
                    variations_per_question: int = 10) -> pd.DataFrame:
        """Main augmentation method - chooses single or multi-GPU"""
        if self.use_multi_gpu and self.num_gpus > 1:
            return self.augment_data_multi_gpu(original_csv_path, target_count, variations_per_question)
        else:
            return self.augment_data_single_gpu(original_csv_path, target_count, variations_per_question)

    def augment_data_single_gpu(self, original_csv_path: str, target_count: int = 200000, 
                               variations_per_question: int = 10) -> pd.DataFrame:
        """Execute single-GPU data augmentation"""
        print(f"Target: Generate {target_count:,} rag_label=YES data")
        
        # Load ALL original data (both YES and NO)
        df = pd.read_csv(original_csv_path)
        original_yes_data = df[df['rag_label'] == 'YES'].copy()
        original_no_data = df[df['rag_label'] == 'NO'].copy()
        original_count = len(original_yes_data)
        
        print(f"Original data:")
        print(f"- YES data: {original_count:,}")
        print(f"- NO data: {len(original_no_data):,}")
        print(f"- Total original: {len(df):,}")
        
        # Auto-detect dataset type
        sample_questions = original_yes_data['question'].head(50).tolist()
        dataset_type = self.detect_dataset_type(sample_questions)
        
        # DEBUG: Check for question duplicates in YES data only
        question_counts = original_yes_data['question'].value_counts()
        duplicates = question_counts[question_counts > 1]
        
        print(f"\nDEBUG: YES data analysis")
        print(f"- Total YES data: {len(original_yes_data)}")
        print(f"- Unique questions: {len(question_counts)}")
        print(f"- Questions with duplicates: {len(duplicates)}")
        
        if len(duplicates) > 0:
            print(f"- Most common question appears {duplicates.iloc[0]} times")
            print(f"- Top duplicate: '{duplicates.index[0]}'")
        
        print(f"Detected dataset type: {dataset_type.upper()}")
        
        # Calculate needed augmentation (only for YES data)
        needed_count = target_count - original_count
        print(f"Data to generate: {needed_count:,}")
        
        if needed_count <= 0:
            print("Target count already reached!")
            # Still combine YES and NO data
            final_df = pd.concat([original_yes_data, original_no_data], ignore_index=True)
            return final_df
        
        # Convert YES data to list WITHOUT deduplication (keep all original data)
        print("DEBUG: Using ALL original YES data (no deduplication)")
        
        original_rows = []
        for idx, row in original_yes_data.iterrows():
            row_dict = row.to_dict()
            row_dict['original_index'] = idx  # Use original DataFrame index as unique identifier
            original_rows.append(row_dict)
        
        print(f"DEBUG: Total YES data to process: {len(original_rows)} rows (all original data)")
        
        # Show first few questions for debugging (may include duplicates)
        print("DEBUG: First 10 questions in YES data:")
        for i, row in enumerate(original_rows[:10]):
            print(f"  {i+1}. [idx:{row['original_index']}] {row['question']}")
        
        # Show examples - sample from different parts for diversity in display only
        sample_questions = []
        total_rows = len(original_rows)
        
        if total_rows >= 3:
            # Sample from beginning, middle, end for example display
            sample_indices = [0, total_rows//3, total_rows*2//3]
            for idx in sample_indices:
                if idx < len(original_rows):
                    question = original_rows[idx]['question']
                    sample_questions.append(question)
        
        print(f"DEBUG: Selected {len(sample_questions)} questions for examples")
        
        should_continue = self.print_examples(sample_questions, dataset_type, num_examples=min(3, len(sample_questions)))
        
        if not should_continue:
            print("Data augmentation cancelled by user.")
            # Still return combined data
            final_df = pd.concat([original_yes_data, original_no_data], ignore_index=True)
            return final_df
        
        print(f"\nStarting augmentation of ALL {len(original_rows):,} YES questions...")
        print(f"Target variations per question: {variations_per_question}")
        print(f"Expected total new questions: ~{len(original_rows) * variations_per_question:,}")
        
        # Generate variations for ALL YES data (including duplicates)
        all_new_data = []
        
        for source_row in tqdm(original_rows, desc="Processing YES questions"):
            if len(all_new_data) >= needed_count:
                break
                
            original_question = source_row['question']
            
            # Generate variations
            variations = self.augment_single_question(
                original_question, 
                dataset_type, 
                variations_per_question
            )
            
            # Create new rows for each variation
            for variation in variations:
                if len(all_new_data) >= needed_count:
                    break
                
                new_row = source_row.copy()
                new_row['question'] = variation
                new_row['dataset'] = 'augmented'
                new_row['data_id'] = f'aug_{len(all_new_data):06d}'
                # Keep rag_label as 'YES'
                
                # Remove temporary original_index
                if 'original_index' in new_row:
                    del new_row['original_index']
                
                all_new_data.append(new_row)
        
        print(f"Generated {len(all_new_data):,} new YES questions")
        
        # Create augmented dataframe (only new YES data)
        augmented_df = pd.DataFrame(all_new_data)
        
        # Combine ALL data: original YES + augmented YES + original NO
        final_df = pd.concat([original_yes_data, augmented_df, original_no_data], ignore_index=True)
        
        print(f"\nFinal results:")
        print(f"- Original YES data: {original_count:,}")
        print(f"- Augmented YES data: {len(augmented_df):,}")
        print(f"- Original NO data: {len(original_no_data):,}")
        print(f"- Total final data: {len(final_df):,}")
        print(f"- Total columns maintained: {len(final_df.columns)}")
        
        return final_df
        """Execute model-based data augmentation"""
        print(f"Target: Generate {target_count:,} rag_label=YES data")
        
        # Load original data
        df = pd.read_csv(original_csv_path)
        original_yes_data = df[df['rag_label'] == 'YES'].copy()
        original_count = len(original_yes_data)
        
        print(f"Original YES data: {original_count:,}")
        
        # Auto-detect dataset type
        sample_questions = original_yes_data['question'].head(50).tolist()
        dataset_type = self.detect_dataset_type(sample_questions)
        print(f"Detected dataset type: {dataset_type.upper()}")
        
        # Calculate needed augmentation
        needed_count = target_count - original_count
        print(f"Data to generate: {needed_count:,}")
        
        if needed_count <= 0:
            print("Target count already reached!")
            return original_yes_data
        
        # Prepare for question-by-question augmentation
        all_new_data = []
        questions_processed = 0
        
        # Convert original data to list for easier iteration
        original_rows = original_yes_data.to_dict('records')
        
        print(f"Using {len(original_rows)} source questions")
        
        # Show examples before full processing - get TRULY diverse questions
        sample_questions = []
        seen_questions = set()
        question_patterns = set()
        
        for row in original_rows:
            question = row['question']
            # Create a pattern by removing specific entities
            pattern = re.sub(r'\b(this|that|the)\s+\w+\b', 'ENTITY', question.lower())
            
            if (question not in seen_questions and 
                pattern not in question_patterns and
                len(question) > 20):  # Avoid very short questions
                sample_questions.append(question)
                seen_questions.add(question)
                question_patterns.add(pattern)
            if len(sample_questions) >= 10:  # Get more options
                break
        
        # Select 3 most diverse questions
        diverse_questions = sample_questions[:3] if len(sample_questions) >= 3 else sample_questions
        
        should_continue = self.print_examples(diverse_questions, dataset_type, num_examples=len(diverse_questions))
        
        if not should_continue:
            print("Data augmentation cancelled by user.")
            return original_yes_data
        
        # Generate variations for each original question
        for source_row in tqdm(original_rows, desc="Processing questions"):
            if len(all_new_data) >= needed_count:
                break
                
            source_question = source_row['question']
            
            # Generate variations for this question
            variations = self.augment_single_question(
                source_question, 
                dataset_type, 
                variations_per_question
            )
            
            # Create new rows for each variation, copying all original data
            for i, variation in enumerate(variations):
                if len(all_new_data) >= needed_count:
                    break
                
                # Copy all original column values
                new_row = source_row.copy()
                
                # Update only the necessary columns
                new_row['question'] = variation
                new_row['dataset'] = 'augmented'
                new_row['data_id'] = f'aug_{len(all_new_data):06d}'
                # Keep rag_label as 'YES' (already copied from source)
                # Keep all other columns (llava_answer, evidence, etc.) from source
                
                all_new_data.append(new_row)
            
            questions_processed += 1
            
            # Break if we've processed enough questions
            if len(all_new_data) >= needed_count:
                break
        
        print(f"Generated {len(all_new_data):,} new questions from {questions_processed} source questions")
        
        # Create augmented dataframe
        augmented_df = pd.DataFrame(all_new_data)
        
        # Combine with original data
        final_df = pd.concat([original_yes_data, augmented_df], ignore_index=True)
        
        print(f"\nFinal results:")
        print(f"- Original YES data: {original_count:,}")
        print(f"- Augmented data: {len(augmented_df):,}")
        print(f"- Total YES data: {len(final_df):,}")
        print(f"- Total columns maintained: {len(final_df.columns)}")
        
        return final_df

    def save_augmented_data(self, df: pd.DataFrame, output_path: str):
        """Save augmented data"""
        df.to_csv(output_path, index=False)
        print(f"Augmented data saved to {output_path}")

# Usage example
if __name__ == "__main__":
    print("="*60)
    print("SINGLE-GPU RAG DATA AUGMENTATION")
    print("="*60)
    
    # Force single GPU usage to avoid complexity
    print("Using single GPU for simplicity and reliability")
    
    # Initialize augmentor in single GPU mode
    augmentor = ModelBasedRAGAugmentor(use_multi_gpu=False)
    
    # Example usage
    augmented_df = augmentor.augment_data(
        '/data/dataset/evqa/evqa_train_judged_answers.csv',
        target_count=180000,
        variations_per_question=15
    )
    
    # Save results
    augmentor.save_augmented_data(augmented_df, '/data/dataset/evqa/single_gpu_augmented_data.csv')
    
    # Show sample results
    print("\nFinal augmented question samples:")
    if 'dataset' in augmented_df.columns:
        sample_questions = augmented_df[augmented_df['dataset'] == 'augmented']['question'].sample(min(10, len(augmented_df))).tolist()
        for i, q in enumerate(sample_questions, 1):
            print(f"{i:2d}. {q}")
    
    print(f"\n✅ Single GPU data augmentation complete!")