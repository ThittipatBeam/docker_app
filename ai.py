from dowload_model import download_model
import torch


def generate_response(model, tokenizer, prompt,max_token=100):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device == 'cuda':
        print('Using CUDA for generation')
    else:
        print('Using CPU for generation')
        
    model.to(device)
    
    input_text = prompt
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    output = model.generate(input_ids, max_length=max_token, num_return_sequences=1)
    
    # Decode and print the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response
    
    
if __name__ == '__main__':
    
    model, tokenizer = download_model("mistralai/Mistral-7B-Instruct-v0.2")
    
    print(generate_response(model, tokenizer, 'Hello can you tell me the capital of Thailand'))
    
    
    
