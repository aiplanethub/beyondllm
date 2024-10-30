from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the tokenizer and model
model_name = "gpt2"  # Replace with the model you want to use
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define input text
input_text = "Hi."

# Encode the input text
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Generate response
with torch.no_grad():
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated text:", generated_text)
