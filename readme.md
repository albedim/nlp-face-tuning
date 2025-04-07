# Install dependencies:
```bash 
pip install -r requirements.txt
```
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

# Install model:
```bash
huggingface-cli download google/gemma-2-2b-it  --local-dir ./models/base/gemma-2-2b-it
```