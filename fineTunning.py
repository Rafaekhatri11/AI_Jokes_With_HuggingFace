from transformers import AutoTokenizer, AutoModelForSeq2SeqLM   
from langchain.prompts import PromptTemplate
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from langchain.prompts import PromptTemplate
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset('json', data_files='E:/Kachori/AI_Jokes_With_HuggingFace/responses.json')

# Initialize model and tokenizer
original_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

def preprocess_function(examples):
    inputs = examples["input_text"]
    targets = examples["target_text"]

    # Prepare the prompt template
    template = ("You are a standup comedian who creates short jokes based on news headlines. "
                "Given the following news article content, generate a humorous joke that is exactly 2 to 3 lines long "
                "and relevant to the original headline. Here is the content: {newsInput}")

    # Generate prompts based on the template
    prompts = [template.format(newsInput=input_text) for input_text in inputs]

    # Tokenize inputs and targets
    model_inputs = tokenizer(prompts, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs['labels'] = labels['input_ids']

    return model_inputs

# Apply the preprocess function to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# print(tokenized_dataset['train']['input_text'])



lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)


peft_model = get_peft_model(original_model, lora_config)


def print_number_of_trainable_model_parameters(model):
    # print("checking", model.named_parameters())
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        # print("checking param.num\n", _,  param.numel())  
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


print(print_number_of_trainable_model_parameters(peft_model))
# # print(print_number_of_trainable_model_parameters(AutoModelForSeq2SeqLM.from_pretrained(model_name ='google/flan-t5-base', torch_dtype=torch.bfloat16)))


output_dir = f'./training-check-points'

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # Higher learning rate than full fine-tuning.
    num_train_epochs=1,
    logging_steps=1,
    max_steps=1    
)
    
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_dataset["train"],
)

peft_trainer.train()

peft_model_path="./PEFT_FineTunned_Model"

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)