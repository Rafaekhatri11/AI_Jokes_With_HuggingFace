from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch

def get_llm_response(newsInput):
    prompt_template = PromptTemplate(
        input_variables = [newsInput],
        template = "You are a standup comedian who creates short jokes based on news headlines. Given the following news article content, generate a humorous joke that is exactly 2 to 3 lines long and relevant to the original headline. Here is the content: {newsInput}"
    )
    rendered_prompt = prompt_template.format(newsInput=newsInput)


    model_name='google/flan-t5-base'

    # Model initialization

    model_initiated = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # Tokenize Model

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(rendered_prompt, return_tensors='pt')

    # print("checking inputs", inputs)

    output = tokenizer.decode(
        model_initiated.generate(
            inputs["input_ids"], 
            max_new_tokens=200,
        )[0], 
        skip_special_tokens=True
    )

    return output
