import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "gpt2"  # Replace with your chosen model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Make sure the model is in evaluation mode
model.eval()

def calculate_perplexity(text):
    # Tokenize input text
    tokens = tokenizer(text, return_tensors='pt')
    
    # Get the model outputs
    with torch.no_grad():
        outputs = model(**tokens, labels=tokens['input_ids'])
    
    # Extract the loss
    loss = outputs.loss
    
    # Calculate perplexity
    perplexity = torch.exp(loss)
    
    return perplexity.item()

# Example usage
text = "This is an example sentence to calculate perplexity."
perplexity_value = calculate_perplexity(text)
print(f"Perplexity: {perplexity_value}")