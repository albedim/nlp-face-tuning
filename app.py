import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="C:\\Users\\alber\OneDrive\Desktop\\folder\models\\base\gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",  # replace with "mps" to run on a Mac device
)

messages = [
    
]

while True:
    print("A")
    message = input()
    messages.append({"role": "user", "content": message})
    outputs = pipe(messages, max_new_tokens=256)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    messages.append({"role": "assistant", "content": assistant_response})
    print(assistant_response)
# Ahoy, matey! I be Gemma, a digital scallywag, a language-slingin' parrot of the digital seas. I be here to help ye with yer wordy woes, answer yer questions, and spin ye yarns of the digital world.  So, what be yer pleasure, eh? 🦜
