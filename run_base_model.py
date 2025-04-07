import sys
import torch
from transformers import pipeline

if len(sys.argv) > 1:
    pipe = pipeline(
        "text-generation",
        model="C:\\Users\\alber\OneDrive\Desktop\\folder\models\\base\gemma-2-2b-it",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda"
    )

    messages = []
    message = " ".join(sys.argv[1:])
    messages.append({"role": "user", "content": message})
    outputs = pipe(messages, max_new_tokens=256)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    messages.append({"role": "assistant", "content": assistant_response})
    print(assistant_response)
else:
    print("No message provided. Please provide a message as a command line argument.")
    sys.exit(1)
