import sys

from datasets import load_dataset
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch


def load_model(model_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def apply_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"]
    )
    return get_peft_model(model, lora_config)


def load_and_tokenize_dataset(tokenizer, dataset_name, max_length=512):
    dataset = load_dataset("json", data_files=f'dataset/{dataset_name}')["train"]

    def tokenize_function(examples):
        input_ids_list, attention_mask_list, labels_list = [], [], []

        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            context = examples["context"][i]

            full_text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{context}<end_of_turn>"
            tokenized = tokenizer(full_text, padding="max_length", truncation=True, max_length=max_length)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            labels = input_ids.copy()

            model_start_text = "<start_of_turn>model\n"
            model_start_tokens = tokenizer(model_start_text, add_special_tokens=False)["input_ids"]

            for j in range(len(input_ids) - len(model_start_tokens) + 1):
                if input_ids[j:j + len(model_start_tokens)] == model_start_tokens:
                    response_start = j + len(model_start_tokens)
                    labels[:response_start] = [-100] * response_start
                    break

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format("torch")
    return tokenized_dataset


def validate_dataset(dataset, tokenizer, expected_length=512):
    example = dataset[0]
    if -100 in example["input_ids"]:
        raise ValueError("input_ids contains -100")
    for key in ["input_ids", "attention_mask", "labels"]:
        if len(example[key]) != expected_length:
            raise ValueError(f"Expected {key} to have length {expected_length}, got {len(example[key])}")
    return True


def generateBenchmark(result, model_name):
    x = list(result.keys())
    y = list(result.values())

    plt.plot(x, y, marker='o')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Simple Line Graph')
    plt.grid(True)
    plt.savefig(f'models/finetuned/{model_name}/benchmark.png', dpi=300)


def train_model(model, tokenizer, tokenized_dataset, model_name, epochs):
    training_args = TrainingArguments(
        output_dir="./training_results",
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./training_logs",
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
        max_grad_norm=0.3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    log_history = trainer.state.log_history
    model.save_pretrained(f"./models/finetuned/{model_name}")
    tokenizer.save_pretrained(f"./models/finetuned/{model_name}")

    result = {}
    for log in log_history:
        if "loss" in log:
            result[log['step']] = log['loss']

    generateBenchmark(result, model_name)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python fine_tune.py <model_path> <fine_tuned_model_name> <dataset_name> <epochs>")
        sys.exit(1)

    model, tokenizer = load_model(sys.argv[1])
    model = apply_lora(model)
    tokenized_dataset = load_and_tokenize_dataset(tokenizer, sys.argv[3])
    validate_dataset(tokenized_dataset, tokenizer)
    train_model(model, tokenizer, tokenized_dataset, sys.argv[2], int(sys.argv[4]))
