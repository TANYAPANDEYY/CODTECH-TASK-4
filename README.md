# CODTECH-TASK-4
GENERATIVE TEXT MODEL

NAME- TANYA PANDEY
COMAPNY- CODTECH SOLUTIONS
ID- CTO8ONH
DOMAIN- AI
DURATION-JANUARY TO FEBRUARY 2025
MENTOR- NEELA SANTOSH KUMAR

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to generate text
def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    output = model.generate(
        input_ids, 
        max_length=max_length, 
        temperature=0.7,  # Controls randomness
        top_k=50,  # Limits the number of next-word options
        top_p=0.95,  # Nucleus sampling
        repetition_penalty=1.2  # Reduces repetitive output
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example prompt
user_prompt = "Once upon a time in a futuristic world,"
generated_text = generate_text(user_prompt)

# Display generated text
print("\nGenerated Text:\n")
print(generated_text)


