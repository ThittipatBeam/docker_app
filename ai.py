from dowload_model import download_model
import torch


model, tokenizer = download_model("mistralai/Mistral-7B-Instruct-v0.2")


# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Encode the input text
input_text = "Hello, how are you?"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Generate a response
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode and print the response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)

