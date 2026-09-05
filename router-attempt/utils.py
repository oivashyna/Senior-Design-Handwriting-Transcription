import re
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from IPython.display import display
from datasets import load_dataset
from typing import Dict, Any, List, Optional, Tuple
import json
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch

pd.options.mode.chained_assignment = None


def load_and_display_nectar(subset: str = "train") -> pd.DataFrame:
    """
    Load a Nectar dataset from Hugging Face and display the first few rows.
    """
    # Load the dataset - fix this option
    dataset = load_dataset("berkeley-nest/Nectar")
    nectar_df = dataset[subset].to_pandas()

    # Display 1 row
    with pd.option_context("display.max_colwidth", None):
        display(nectar_df.head(1))

    # Compute the number of queries with GPT-4 responses
    nectar_df_expanded = nectar_df.explode("answers")
    nectar_df_expanded["model"] = nectar_df_expanded["answers"].apply(
        lambda x: x["model"]
    )
    print(
        f"Number of queries with GPT-4 responses: {len(nectar_df_expanded[nectar_df_expanded['model'] == 'gpt-4'])}"
    )
    return nectar_df


def preprocess_nectar(
    df: pd.DataFrame, model: str, response_column: str
) -> pd.DataFrame:
    """
    Specific preprocessing of the Nectar dataset.
    """
    # Filter to include only the first turn, good-natured responses, and responses that contain the specified model - swap for more appropriate columns
    conditions = (
        (df["turns"] == 1)
        & df["good_natured"]
        & df["answers"].apply(lambda ans: any(model == a.get("model") for a in ans))
    )
    filtered_df = df[conditions].copy()

    # Extract the answer for the specified model from the list of all answers
    filtered_df[response_column] = filtered_df["answers"].apply(
        lambda row: next(
            (item["answer"] for item in row if item["model"] == model), None
        )
    )

    # Clean user prompts
    pattern_start = re.compile(r"^\s+[Hh]uman:\s+")
    pattern_end = re.compile(r"\s+[Aa]ssistant:\s+$")

    filtered_df["prompt"] = filtered_df["prompt"].apply(
        lambda prompt: pattern_end.sub("", pattern_start.sub("", prompt)).strip()
    )

    # Drop unnecessary columns
    filtered_df.drop(
        columns=["answers", "num_responses", "turns", "good_natured"], inplace=True
    )

    return filtered_df

#either keep the same or change for the sake of studying other chat format for judging
def to_gemini_api_messages(
    messages: List[str], system_message: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """Convert the conversation to Gemini chat completion format."""
    system_instruction = system_message if system_message else "You are a helpful assistant."
    
    contents = []
    for i, turn in enumerate(messages):
        # Gemini uses "user" and "model"
        role = "user" if i % 2 == 0 else "model"
        
        # Gemini expects a list of "parts" containing the text
        contents.append({
            "role": role,
            "parts": [{"text": turn}]
        })
        
    return contents, system_instruction

def prepare_llm_queries(
    dataset_df: pd.DataFrame, system_message: Optional[str] = None
) -> Dict[int, List[Dict[str, str]]]:
    """Prepare queries for using LLM endpoints"""
    queries = {}
    for pidx, row in dataset_df.to_dict(orient="index").items():
        prompt = row["prompt"]
        if type(prompt) == str:
            prompt = [prompt]
        messages = to_gemini_api_messages(prompt, system_message) #potentially change for gemini format
        queries[pidx] = messages
    return queries

def format_judge_prompt(
    judge_template: Dict[str, Any], question: str, answer: str, reference: str
) -> str:
    """Format the prompt for the judge endpoint."""
    return judge_template["prompt_template"].format(
        instruction=judge_template["instruction"],
        question=question,
        answer=answer,
        ref_answer_1=reference,
    )

def prepare_llm_judge_queries(
    dataset_df: pd.DataFrame,
    judge_template: Dict[str, Any],
    answer_key: str,
    reference_key: str,
) -> Dict[int, List[Dict[str, str]]]:
    """Prepare queries for using LLM judge endpoint"""
    queries = {}
    for pidx, row in dataset_df.to_dict(orient="index").items():
        prompt = format_judge_prompt(
            judge_template, row["prompt"], row[answer_key], row[reference_key]
        )
        messages = to_gemini_api_messages([prompt]) #again potentially change for gemini format
        queries[pidx] = messages
    return queries

def inspect_llm_judge_queries(
    dataset_df: pd.DataFrame,
    template_path="assets/judge_template.json",
    #change both below to allow for multiple answers
    answer_key="mixtral_response",
    reference_key="gpt4_response",
):
    """Inspect one prompt from the prepared LLM judge queries"""
    with open(template_path) as f:
        judge_template = json.load(f)

    example_row = dataset_df.iloc[4]
    prompt = format_judge_prompt(
        judge_template,
        example_row["prompt"],
        example_row[answer_key],
        example_row[reference_key],
    )
    print(prompt)

def parse_judge_responses(
    judge_responses: Dict[int, str]
) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Parses the llm-judge responses and extracts the labels and explanations.
    """
    labels, explanations = {}, {}
    for pidx, response in judge_responses.items():
        match = re.search(r"\[\[([\d\.]+)\]\]\n(.+)", response)
        if match:
            score, explanation = int(float(match.group(1))), match.group(2)
        else:
            score, explanation = -1, ""

        labels[pidx] = score
        explanations[pidx] = explanation
    return labels, explanations

#is it best to have it as
def visualize_label_distribution(dataset_df: pd.DataFrame, key: str) -> None:
    """
    Visualizes the label distribution of a dataset.
    """
    # Create a counter for the label distribution
    dataset_counter = Counter(dataset_df[key])

    # Plot the bar chart
    plt.bar(dataset_counter.keys(), dataset_counter.values())
    plt.xlabel(key.capitalize())
    plt.ylabel("Count")
    plt.title(f"Histogram of {key}")
    plt.xticks(list(dataset_counter.keys()))
    plt.show()

def balance_dataset(
    dataset_df: pd.DataFrame, key: str, random_state: int = 42
) -> pd.DataFrame:
    """
    Balance the dataset by oversampling the minority class.
    """
    # Determine the minority class
    min_count = dataset_df[key].value_counts().min()

    # Create a balanced DataFrame
    sampled_dfs = []
    for label in dataset_df[key].unique():
        sampled = dataset_df[dataset_df[key] == label].sample(
            n=min_count, random_state=random_state
        )
        sampled_dfs.append(sampled)

    balanced_df = pd.concat(sampled_dfs).sample(frac=1, random_state=random_state)
    return balanced_df

#what i realized here: there is no actual training that occurs here - change to make more useful 
def train(self):
        """
        Train the causal LM using HuggingFace Trainer.
        """
        # Prepare dataset
        print("[CausalLMTrainer] Preparing dataset...")
        train_dataset = self._prepare_dataset()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.save_model_path,
            num_train_epochs=self.model_config.get("num_epochs", 3),
            per_device_train_batch_size=self.model_config.get("batch_size", 4),
            gradient_accumulation_steps=self.model_config.get("gradient_accumulation_steps", 4),
            learning_rate=self.model_config.get("learning_rate", 2e-5),
            weight_decay=self.model_config.get("weight_decay", 0.01),
            warmup_ratio=self.model_config.get("warmup_ratio", 0.1),
            logging_steps=self.model_config.get("logging_steps", 10),
            save_steps=self.model_config.get("save_steps", 100),
            save_total_limit=2,
            fp16=torch.cuda.is_available() and self.model_config.get("fp16", True),
            report_to=self.model_config.get("report_to", "none"),
            remove_unused_columns=False
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )

        # Train
        print("[CausalLMTrainer] Starting training...")
        trainer.train()

        # Save model
        print(f"[CausalLMTrainer] Saving model to {self.save_model_path}")
        trainer.save_model(self.save_model_path)
        self.tokenizer.save_pretrained(self.save_model_path)

        # Merge LoRA weights if using LoRA
        if self.model_config.get("use_lora", True) and self.model_config.get("merge_lora", True):
            self._merge_and_save_lora()

        print("[CausalLMTrainer] Training completed!")

def prepare_ft_messages(dataset_df: pd.DataFrame, label_key: str) -> pd.DataFrame:
    """
    Add messages for fine-tuning using the dataset dataframe, system message, and classifier message.
    """
    with open(f"assets/system_ft.txt", "r") as f1, open(
        f"assets/classifier_ft.txt", "r"
    ) as f2:
        system_message = f1.read()
        classifier_message = f2.read()

    # Create API formatted 'messages' column for each row in the dataset dataframe
    return dataset_df.apply(
        lambda row: to_gemini_api_messages(
            [
                classifier_message.format(question=row["prompt"]),
                f"[[{row[label_key]}]]",
            ],
            system_message,
        ),
        axis=1,
    )
def inspect_instructions() -> None:
    """
    Inspect the instructions used for instruction fine-tuning.
    """
    with open(f"assets/system_ft.txt", "r") as f1, open(
        f"assets/classifier_ft.txt", "r"
    ) as f2:
        system_message = f1.read()
        classifier_message = f2.read()

    print("\n".join([system_message, classifier_message]))

def update_yaml_with_env_vars(file_path, env_vars):
    """
    Updates the YAML file at file_path with the given environment variables.
    """
    with open(file_path) as file:
        yaml_content = yaml.safe_load(file)

    yaml_content["env_vars"] = env_vars

    with open(file_path, "w") as file:
        yaml.dump(yaml_content, file)