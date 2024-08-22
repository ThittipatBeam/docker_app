from dotenv import load_dotenv
import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

hf_token = os.getenv('HF_TOKEN')

def download_model(model_name):
    
    login(token=hf_token,add_to_git_credential=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer


# if __name__ == "__main__":
    
#     model, tokenizer = download_model("mistralai/Mistral-7B-Instruct-v0.2")


