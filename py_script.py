
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig


from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")
model = AutoModelForCausalLM.from_pretrained("sarvamai/sarvam-1")


dataset = load_dataset("json", data_files="hindi_wiki_llm_pairs.jsonl")




tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(data):
    print(data)
    input = data['source']
    output = data['target']
    tokenized_inputs = tokenizer(input, truncation=True, padding='max_length', max_length=128)
    tokenized_outputs = tokenizer(output, truncation=True, padding='max_length', max_length=128)
    tokenized_inputs['labels'] = tokenized_outputs['input_ids']

    return tokenized_inputs




tokenized_dataset = dataset['train'].map(tokenize_function)



tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)




lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)


from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./finetuned-hindi",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2, 
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=2,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="none",
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    # data_collator=data_collator
)

# Start fine-tuning
trainer.train()

# Save fine-tuned model
trainer.save_model("./finetuned-hindi-model")
tokenizer.save_pretrained("./finetuned-hindi-model")


