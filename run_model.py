import json
import sys
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def check_args():
    modelPath, maxTokens = "", 256
    if len(sys.argv) < 2:
        print("Usage: python run_model.py <model_path> [max_tokens]")
        sys.exit(1)
    modelPath = sys.argv[1]
    if len(sys.argv) > 2:
        maxTokens = int(sys.argv[2])
    return modelPath, maxTokens, "base" in modelPath

if __name__ == "__main__":
    modelPath, maxTokens, base = check_args()

    result = []
    with open("test/questions.txt", "r") as f:
        for line in f:
            result.append({
                "prompt": line.strip(),
                "response": ""
            })

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(modelPath)
    model = AutoModelForCausalLM.from_pretrained(
        modelPath,
        quantization_config=quantization_config,
    )

    for i in range(len(result)):
        formatted_prompt = f"<start_of_turn>user\n{result[i]['prompt']}<end_of_turn>\n<start_of_turn>model\n"
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**input_ids, max_new_tokens=maxTokens)
        print(f"Processing prompt {i + 1}/{len(result)}")
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        response_start = full_response.find("<start_of_turn>model\n") + len("<start_of_turn>model\n")
        response_end = full_response.find("<end_of_turn>", response_start)

        if response_end == -1:
            model_response = full_response[response_start:]
        else:
            model_response = full_response[response_start:response_end]

        result[i]['response'] = model_response.strip()

    fileName = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if base:
        with open(f"test/result/base/{fileName}.json", "w") as f:
            json.dump(result, f, indent=2)
    else:
        with open(f"test/result/finetuned/{fileName}.json", "w") as f:
            json.dump(result, f, indent=2)