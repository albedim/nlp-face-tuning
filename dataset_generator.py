import os
import re
import json
import random

with open("dataset/raw_dataset.txt", "r", encoding="utf-8") as file:
    # lines = file.readlines()[1:]
    lines = file.readlines()

"""
pattern = r"\[\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2}:\d{2}\s?[APap][Mm]?\] [^:]+: (.+)"

messages = []
for line in lines:
    match = re.match(pattern, line.strip())
    if match:
        messages.append(match.group(1))"""

# lines->messages
output = "\n".join(lines)
"""
output = output.replace("This message was deleted.", "")
output = output.replace(".", "")
output = output.replace("‎", "")"""


with open("dataset/temp.txt", "w", encoding="utf-8") as output_file:
    output_file.write(output)

user_prompts = [
    "Cosa ne pensi di questo?",
    "Puoi spiegarmi questo?",
    "Come affronti solitamente questa situazione?",
    "Sei d'accordo o in disaccordo?",
    "Cosa faresti in questo caso?",
    "Potresti spiegare meglio?",
    "Perché pensi che succeda questo?",
    "Hai dei consigli?",
    "Come la vedi?",
    "Qual è il tuo punto di vista?",
    "Pensi che sia una buona idea?",
    "Cosa consiglieresti?",
    "Cosa ne pensi?",
    "Puoi elaborare su questo?",
    "Qual è il tuo approccio?",
    "Pensi che funzioni?",
    "Come affronti questo tipo di situazioni?",
    "Spiega questo come se fossi alle prime armi.",
    "Hai mai provato qualcosa del genere?",
    "Lo consiglieresti?"
]

with open('dataset/temp.txt', "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

dataset = []
for line in lines:
    prompt = random.choice(user_prompts)
    sample = {
        "instruction": prompt,
        "context": line
    }
    dataset.append(sample)

with open('dataset/fine_tune_dataset.jsonl', "w", encoding="utf-8") as f:
    for entry in dataset:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

os.remove("dataset/temp.txt")
print(f"✅ Dataset created with {len(dataset)} sets of data → saved in {output_file}")