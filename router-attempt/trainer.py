#may need to fix module structuring here
import os
from os import listdir
import yaml
import json
import re
import numpy as np
import generate_prompt
import subprocess
#import model_loader
#import torch.nn as nn
import copy
from sklearn.svm import SVC
from PIL import Image
#from churro_ocr.ocr import OCRClient
#from churro_ocr.providers import OCRBackendSpec, build_ocr_backend
import image_transcriber
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from sklearn.ensemble import VotingClassifier
from botocore.exceptions import NoCredentialsError, ClientError, BotoCoreError
#fix this somehow in order to set up text rather than list

#same for this thing
#def invoke_vertex_model(client_and_model: tuple, image_data_list: list, prompt: str) -> dict:
# data setup here
#def load_dataset():

from datasets import load_dataset
# check for only english - also need to figure out pandas combo

#potentially only use these last two but there might be a point in using the train version as well
test_ds = load_dataset("stanford-oval/churro-dataset", split="test")
dev_ds = load_dataset("stanford-oval/churro-dataset", split="dev")
dev_ds_Dt = dev_ds.to_pandas()
print(dev_ds_Dt.columns)
test_ds_Dt = test_ds.to_pandas()
dev_ds_Dt = dev_ds_Dt.drop(columns={'dataset_id', 'languages', 'main_script', 'original_transcription', 'scripts'})
dev_ds_Dt = dev_ds_Dt.rename(columns={"cleaned_transcription": "transcription"})
test_ds_Dt =test_ds_Dt.drop(columns={'dataset_id', 'languages', 'main_script', 'original_transcription', 'scripts' })
test_ds_Dt =test_ds_Dt.rename(columns={"cleaned_transcription": "transcription"})

#filter for english, german, dutch

transcript_info = pd.read_csv("newberry-transcriptions20260407.csv")
transcript_info = transcript_info.drop(columns={'title','permalink', 'translation'})
transcript_info["filename"] = transcript_info["filename"].str.split(".", n=1).str[0]
transcript_info =transcript_info.merge(dev_ds_Dt,left_on="transcription",right_on="transcription")
transcript_info=transcript_info.merge(test_ds_Dt,left_on="transcription",right_on="transcription")
transcript_info=transcript_info.filter(items=["English", "German", "Dutch"])
transcript_info['gemini_trans'] = np.nan
transcript_info['claude_trans'] = np.nan
transcript_info['gpt_trans'] = np.nan
transcript_info['churro_trans'] = np.nan
transcript_info['kraken_trans'] = np.nan
transcript_info['tesseract_trans'] = np.nan


transcript_info['gemini_judge'] = {"cer": np.nan, "wer": np.nan, "similarity": np.nan}
transcript_info['gemini_judge_gpt'] = np.nan
transcript_info['gemini_judge_claude'] = np.nan
transcript_info['gemini_judge_churro'] = np.nan
transcript_info['claude_judge'] = {"cer": np.nan, "wer": np.nan, "similarity": np.nan}
transcript_info['claude_judge_gpt'] = np.nan
transcript_info['claude_judge_gemini'] = np.nan
transcript_info['claude_judge_churro'] = np.nan
transcript_info['gpt_judge'] = {"cer": np.nan, "wer": np.nan, "similarity": np.nan}
transcript_info['gpt_judge_gemini'] = np.nan
transcript_info['gpt_judge_claude'] = np.nan
transcript_info['gpt_judge_churro'] = np.nan
transcript_info['churro_judge'] = {"cer": np.nan, "wer": np.nan, "similarity": np.nan}

transcript_info['final_judge'] = np.nan #use this to determine model name
#potentially change to be specifically the list of items in 
#figure out profile
#folder_dir = "C:/Users/lenab/OneDrive/Pictures/Screenshots/school/cop4934/router-attempt/2026Jul22 - 836 files"
folder_dir = "/home/handwritingTranscriptSD/Desktop/Handwriting-Development/historical-image-transcription-script/router-attempt/2026Jul22 - 836 files"
for images in os.listdir(folder_dir):
   
    try:
        image_path = os.path.join(folder_dir, images)
        image = Image.open(image_path)
        image.load()
        
        image_name = os.path.splitext(images)[0]
        image_info = image_transcriber.load_and_encode_image(image_name)
        gptclient =    image_transcriber.create_bedrock_client(None, 'us-east-1')
        #add access to prompt later
        prompt1 = subprocess.run(["uv", "run","python", "generate_prompt.py", "--model", "gpt"]) 
        #change to be converse
        gptinvoke = image_transcriber.invoke_bedrock_model(gptclient, 'gpt', image_info,prompt1 )
        gptresponse = image_transcriber.format_output(gptinvoke, 'gpt', image_info)
        

        claudeclient = image_transcriber.create_bedrock_client(None, 'us-east-1')
        prompt2 = subprocess.run(["uv", "run","python", "generate_prompt.py", "--model", "claude"]) 
        claudeinvoke =image_transcriber.invoke_bedrock_model(claudeclient,'us.anthropic.claude-sonnet-4-5-20250929-v1:0', image_info, prompt2)
        clauderesponse = image_transcriber.format_output(claudeinvoke, 'us.anthropic.claude-sonnet-4-5-20250929-v1:0', image_info)
       
             
        geminiclient = image_transcriber.create_vertex_client(project_id,'us-central1', 'gemini-3.6-flash' )
        prompt3 = subprocess.run(["uv", "run","python", "generate_prompt.py", "--model", "gemini"]) 
        geminiinvoke =  image_transcriber.create_vertex_client(geminiclient, 'gemini-3.6-flash', image_info, prompt3 )
        geminiresponse = image_transcriber.format_output(geminiinvoke, 'gemini-3.6-flash',image_info )
        
        
             
        #churro attempt
        backend = build_ocr_backend(OCRBackendSpec(
        provider="hf",
        model="stanford-oval/churro-3B",
    )
)
#obv change image path
        page = OCRClient(backend).ocr_image(image_path=images)
     
        condition = transcript_info["filename"] == image_name
        if condition.any():
             transcript_info.loc[condition, "gpt_trans"] = gptresponse
             transcript_info.loc[condition, "claude_trans"] = clauderesponse
             transcript_info.loc[condition, "gemini_trans"] = geminiresponse
             transcript_info.loc[condition, "churro_trans"] = page.text
        else:
             new_row = {
                   "filename": image_name,
                    "gpt_trans": gptresponse,
                    "claude_trans": clauderesponse,
                    "gemini_trans": geminiresponse,
                    "churro_trans": page.text,
                    "kraken_trans": np.nan,
                    "tesseract_trans": np.nan
    }

             transcript_info = pd.concat(
                 [transcript_info, pd.DataFrame([new_row])],ignore_index=True)

             
  #def change error message
    except FileNotFoundError:
            # Re-raise with more context
            continue
            raise FileNotFoundError(f"Image file not found: {images}")
            
    except PermissionError:
            raise PermissionError(
                f"Permission denied: Cannot read image file '{images}'. "
                "Please check your file permissions."
            )
    except Image.UnidentifiedImageError:
            raise ValueError(
                f"Cannot identify image file '{images}'. "
                f"The file may be corrupted or not a valid image. "
                f"Supported formats: PNG, JPEG, JPG"
            )
    except OSError as e:
            raise OSError(f"Failed to open image file '{images}': {str(e)}")
    except Exception as e:
            raise ValueError(f"Failed to load image '{images}': {str(e)}")

    #combine images with their actual transcription

for row in dev_ds_Dt:
     image = Image.open(row["example_id"])
     image.load()
     image_name = row["example_id"]
     image_info = image_transcriber.load_and_encode_image(image_name)
     gptclient =    image_transcriber.create_bedrock_client(None, 'us-east-1')
             #add access to prompt later
     prompt1 = subprocess.run(["uv", "run","python", "generate_prompt.py", "--model", "gpt"]) 
     gptinvoke = image_transcriber.invoke_bedrock_model(gptclient, 'gpt', image_info,prompt1 )
     gptresponse = image_transcriber.format_output(gptinvoke, 'gpt', image_info)
             
     
     claudeclient = image_transcriber.create_bedrock_client(None, 'us-east-1')
     prompt2 = subprocess.run(["uv", "run","python", "generate_prompt.py", "--model", "claude"]) 
     claudeinvoke =image_transcriber.invoke_bedrock_model(claudeclient,'us.anthropic.claude-sonnet-4-5-20250929-v1:0', image_info, prompt2)
     clauderesponse = image_transcriber.format_output(claudeinvoke, 'us.anthropic.claude-sonnet-4-5-20250929-v1:0', image_info)
            
                  
     geminiclient = image_transcriber.create_vertex_client(project_id,'us-central1', 'gemini-3.6-flash' )
     prompt3 = subprocess.run(["uv", "run","python", "generate_prompt.py", "--model", "gemini"]) 
     geminiinvoke =  image_transcriber.create_vertex_client(geminiclient, 'gemini-3.6-flash', image_info, prompt3 )
     geminiresponse = image_transcriber.format_output(geminiinvoke, 'gemini-3.6-flash',image_info )
             
             
                  
             #churro attempt
     backend = build_ocr_backend(OCRBackendSpec(
        provider="hf",
             model="stanford-oval/churro-3B",
         )
     )
     #obv change image path
     page = OCRClient(backend).ocr_image(row["image"])
     
          
     condition = transcript_info["image"] == image_name
     if condition.any():
                  transcript_info.loc[condition, "gpt_trans"] = gptresponse
                  transcript_info.loc[condition, "claude_trans"] = clauderesponse
                  transcript_info.loc[condition, "gemini_trans"] = geminiresponse
                  transcript_info.loc[condition, "churro_trans"] = page.text
     else:
                  new_row = {
                        
                         "gpt_trans": gptresponse,
                         "claude_trans": clauderesponse,
                         "gemini_trans": geminiresponse,
                         "churro_trans": page.text,
                         "kraken_trans": np.nan,
                         "tesseract_trans": np.nan
         }
     
     transcript_info = pd.concat(
                      [transcript_info, pd.DataFrame([new_row])],ignore_index=True)
     
for row in test_ds_Dt:
     image = Image.open(row["image"])
     image.load()
     image_info = image_transcriber.load_and_encode_image(image)
     image_name = row["example_id"]
     gptclient =    image_transcriber.create_bedrock_client(None, 'us-east-1')
             #add access to prompt later
     prompt1 = subprocess.run(["uv", "run","python", "generate_prompt.py", "--model", "gpt"]) 
     gptinvoke = image_transcriber.invoke_bedrock_model(gptclient, 'gpt', image_info,prompt1 )
     gptresponse = image_transcriber.format_output(gptinvoke, 'gpt', image_info)
             
     
     claudeclient = image_transcriber.create_bedrock_client(None, 'us-east-1')
     prompt2 = subprocess.run(["uv", "run","python", "generate_prompt.py", "--model", "claude"]) 
     claudeinvoke =image_transcriber.invoke_bedrock_model(claudeclient,'us.anthropic.claude-sonnet-4-5-20250929-v1:0', image_info, prompt2)
     clauderesponse = image_transcriber.format_output(claudeinvoke, 'us.anthropic.claude-sonnet-4-5-20250929-v1:0', image_info)
            
                  
     geminiclient = image_transcriber.create_vertex_client(project_id,'us-central1', 'gemini-3.6-flash' )
     prompt3 = subprocess.run(["uv", "run","python", "generate_prompt.py", "--model", "gemini"]) 
     geminiinvoke =  image_transcriber.create_vertex_client(geminiclient, 'gemini-3.6-flash', image_info, prompt3 )
     geminiresponse = image_transcriber.format_output(geminiinvoke, 'gemini-3.6-flash',image_info )
             
             
                  
             #churro attempt
     backend = build_ocr_backend(OCRBackendSpec(
        provider="hf",
             model="stanford-oval/churro-3B",
         )
     )
     #obv change image path
     page = OCRClient(backend).ocr_image(row["image"])
     
          
     condition = transcript_info["filename"] == image_name
     if condition.any():
                  transcript_info.loc[condition, "gpt_trans"] = gptresponse
                  transcript_info.loc[condition, "claude_trans"] = clauderesponse
                  transcript_info.loc[condition, "gemini_trans"] = geminiresponse
                  transcript_info.loc[condition, "churro_trans"] = page.text
     else:
                  new_row = {
                       
                         "gpt_trans": gptresponse,
                         "claude_trans": clauderesponse,
                         "gemini_trans": geminiresponse,
                         "churro_trans": page.text,
                         "kraken_trans": np.nan,
                         "tesseract_trans": np.nan
         }
     
     transcript_info = pd.concat(
                      [transcript_info, pd.DataFrame([new_row])],ignore_index=True)               
      
      


#e version with ground truth - call accuracy here
condition = transcript_info["transcription"].notna()
#go back to add image
judge_question = "Evaluate the transcription that was given by each LLM and rate them on a scale of 1 to 5, 1 being worst and 5 being best. Note that these are transcriptions of documents from the 1680s and 1730s and therefore will have historical writing"
for column_name, item in transcript_info.items():
    if condition.any():
     gpt_judge = image_transcriber.calculate_accuracy(item["gpt_trans"], item["transcription"])
     claude_judge = image_transcriber.calculate_accuracy(item["claude_trans"], item["transcription"])
     gemini_judge = image_transcriber.calculate_accuracy(item["gemini_trans"], item["transcription"])
     churro_judge = image_transcriber.calculate_accuracy(item["churro_trans"], item["transcription"])
     item["gpt_judge"] = gpt_judge
     item["claude_judge"] = claude_judge
     item["gemini_judge"] = gemini_judge
     item["churro_judge"] = churro_judge
     judge_min = min(item["gpt_judge"]["cer"], item["claude_judge"]["cer"], item["gemini_judge"]["cer"], item["churro_judge"]["cer"])
     if(item["gpt_judge"]["cer"] == judge_min):
          item["final_judge"] = "ChatGPT: " + item['gpt_trans']
     elif(item["claude_judge"]["cer"] == judge_min):
          item["final_judge"] = "Claude: " + item['claude_trans']
     elif(item["gemini_judge"]["cer"] == judge_min):
          item["final_judge"] = "Gemini: " + item['gemini_trans']
     else:
          item["final_judge"] = "CHURRO: " + item['churro_trans']
          

    else:
        #fix for later with the invoke functions that were pasted above
        gptclient2 =    image_transcriber.create_bedrock_client(None, 'us-east-1')
        gptinvoke_claude = image_transcriber.invoke_bedrock_model(gptclient, 'gpt', image_info, judge_question + item["claude_trans"] )
        item["gpt_judge_claude"]  =  gptinvoke_claude["judgment"]
        gptinvoke_gemini  = image_transcriber.invoke_bedrock_model(gptclient, 'gpt', image_info,judge_question+ item["gemini_trans"] )
        item["gpt_judge_gemini"]  = gptinvoke_gemini ["judgment"]
        gptinvoke_churro = image_transcriber.invoke_bedrock_model(gptclient, 'gpt',image_info, judge_question +item["churro_trans"])
        item["gpt_judge_churro"]  = gptinvoke_churro["judgment"] #new judge prompt
                        
        claudeclient2 = image_transcriber.create_bedrock_client(None, 'us-east-1')
        claudeinvoke_gpt = image_transcriber.invoke_bedrock_model(claudeclient,'us.anthropic.claude-sonnet-4-5-20250929-v1:0',image_info, judge_question+item["gpt_trans"])
        item["claude_judge_gpt"]  = claudeinvoke_gpt["judgment"]
        claudeinvoke_gemini = image_transcriber.invoke_bedrock_model(claudeclient,'us.anthropic.claude-sonnet-4-5-20250929-v1:0',image_info, judge_question+item["gemini_trans"])
        item["claude_judge_gemini"]  = claudeinvoke_gemini["judgment"]
        claudeinvoke_churro = image_transcriber.invoke_bedrock_model(claudeclient,'us.anthropic.claude-sonnet-4-5-20250929-v1:0',image_info, judge_question+item["churro_trans"])
         #someway to establish a vote here
        item["claude_judge_churro"]  = claudeinvoke_churro["judgment"]
                                  
        geminiclient2 = image_transcriber.create_vertex_client(project_id,'us-central1', 'gemini-3.6-flash' )
        geminiinvoke_gpt =  image_transcriber.create_vertex_client(geminiclient, 'gemini-3.6-flash',image_info , judge_question+item["gpt_trans"] )
        item["gemini_judge_gpt"]  = geminiinvoke_gpt["judgment"]
        geminiinvoke_claude =  image_transcriber.create_vertex_client(geminiclient, 'gemini-3.6-flash', image_info, judge_question+item["claude_trans"])
        item["gemini_judge_claude"]  = geminiinvoke_claude["judgment"]
        geminiinvoke_churro =  image_transcriber.create_vertex_client(geminiclient, 'gemini-3.6-flash', image_info, judge_question+item["churro_trans"] )
        item["gemini_judge_churro"]  = geminiinvoke_churro["judgment"]

        judge_max = max(item["gemini_judge_churro"] ,  item["gemini_judge_claude"], item["gemini_judge_gpt"], item["claude_judge_churro"],  item["claude_judge_gemini"], item["claude_judge_gpt"], item["gpt_judge_churro"],  item["gpt_judge_claude"], item["gpt_judge_gemini"])
        if(judge_max == item["gemini_judge_churro"] or judge_max == item["gpt_judge_churro"] or judge_max == item["claude_judge_churro"] ):
             item["final_judge"] = "CHURRO: " + item["churro_trans"]
        elif(judge_max ==  item["gpt_judge_claude"]  or item["gemini_judge_claude"]):
             item["final_judge"] = "Claude: " + item["claude_trans"]
        elif(judge_max ==  item["gpt_judge_gemini"]  or item["claude_judge_gemini"]):
             item["final_judge"] = "Gemini: " + item["gemini_trans"]
        else:
              item["final_judge"] = "ChatGPT: " + item["gpt_trans"]

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
#study the causal llm router for better implementation
def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["full_text"],
                truncation=True,
                max_length=self.model_config.get("max_length", 512),
                padding="max_length"
            )

tokenized_dataset = transcript_info.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)
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
#save
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

#potentially fix in path section
def update_yaml_with_env_vars(file_path, env_vars):
    """
    Updates the YAML file at file_path with the given environment variables.
    """
    with open(file_path) as file:
        yaml_content = yaml.safe_load(file)

    yaml_content["env_vars"] = env_vars

    with open(file_path, "w") as file:
        yaml.dump(yaml_content, file)


