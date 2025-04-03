import os 
import torch
import argparse
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from datasets import Dataset
from transformers import TrainingArguments
import wandb
import pandas as pd
# from sklearn.model_selection import train_test_split
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
WB_TOKEN = os.getenv("WB_TOKEN")
DATASET_URL = os.getenv("DATASET_URL")

def setup_wandb(model_type):
    hf_secret = HF_TOKEN
    wandb_secret = WB_TOKEN
    login(token=hf_secret)
    print("huggingface_hub installed successfully!")
    wandb.login(key=wandb_secret)
    run = wandb.init(
        project=f"qwen2.5-{model_type}b-finetune-unsloth",
        config={"learning_rate": 4e-4},
    )
    return run

def load_model(model_type):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"unsloth/Qwen2.5-{model_type}B-unsloth-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
        use_rslora=True,
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer



def formatting_prompts_func(examples,tokenizer):
    EOS_TOKEN = tokenizer.eos_token
    alpaca_prompt = """Dưới đây là một hướng dẫn mô tả nhiệm vụ, kèm theo một đầu vào cung cấp thêm ngữ cảnh. Hãy viết một phản hồi hoàn chỉnh để đáp ứng yêu cầu.

    ### Hướng dẫn:
    {}

    ### Đầu vào:
    {}

    ### Phản hồi:
    {}"""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Concatenate instruction, input, and output with newline characters
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    
    return {"text": texts}

def load_dataset_and_prepare(tokenizer):
    url = DATASET_URL
    df = pd.read_csv(url)
    # df, _ = train_test_split(df, test_size=0.9, random_state=42)
    df = df.iloc[:, 1:].drop_duplicates()
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: formatting_prompts_func(x, tokenizer), batched=True)
    return dataset.select_columns(["text"])

def train_model(model, dataset, batch_size):    
    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        args = TrainingArguments(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 10, 
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "wandb", 
        ),
    )
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Train the model with specific batch size.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--model_type", type=str, default="3", help="Model Type  for training such as 3B, 7B,...")
    args = parser.parse_args()

    print(f"Starting training with batch size: {args.batch_size}")
    run = setup_wandb(args.model_type)
    model, tokenizer = load_model(args.model_type)
    dataset = load_dataset_and_prepare(tokenizer)
    trainer = train_model(model, dataset, args.batch_size)
    trainer.train()
    model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
    run.finish()

if __name__ == "__main__":
    main()

# python D:\DataScience_For_mySelf\Graduation_thesis\RAG-Viverse\src\rag-viverse\finetuning\test_train.py
# D:\DataScience_For_mySelf\Graduation_thesis\RAG-Viverse\src\rag-viverse\finetuning\test_train.py
