# DocString Generator — TinyLlama Fine-Tuning

A fine-tuned language model that generates Python docstrings for a given function's code. Built by fine-tuning **TinyLlama-1.1B-Chat** using **LoRA (SFT)** on the **CodeSearchNet** dataset.

🔗 **Model on Hugging Face Hub:** [ubaid0405/tinyllama-1.1b-docstring-gen](https://huggingface.co/ubaid0405/tinyllama-1.1b-docstring-gen)

---

## Overview

This project fine-tunes a small open-source LLM to automatically generate docstrings for Python functions, given only their code. The goal was to explore parameter-efficient fine-tuning (LoRA) on a lightweight base model that can run on consumer/free-tier GPUs (Google Colab T4).

## Model Details

| | |
|---|---|
| **Base model** | [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| **Method** | Supervised Fine-Tuning (SFT) with LoRA |
| **Dataset** | [code_search_net](https://huggingface.co/datasets/code_search_net) (Python subset) |
| **Framework** | 🤗 Transformers, TRL (`SFTTrainer`), PEFT |
| **Training environment** | Google Colab (T4 GPU) |

## Training Results

| Epoch | Training Loss | Validation Loss | Entropy | Mean Token Accuracy |
|-------|---------------|------------------|---------|----------------------|
| 1     | 1.165         | 1.195            | 1.175   | 72.76%               |

Training and validation loss stayed close, indicating the model wasn't overfitting given the single-epoch, LoRA-based fine-tuning setup.

## Prompt Format

The model was trained using the following instruction format:

```
[INST] Write a docstring:
<function code> [/INST]
```

## How to Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(base_model, "ubaid0405/tinyllama-1.1b-docstring-gen")
tokenizer = AutoTokenizer.from_pretrained("ubaid0405/tinyllama-1.1b-docstring-gen")

code = "def add(a, b):\n    return a + b"
prompt = f"[INST] Write a docstring:\n{code} [/INST]"

ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
out = model.generate(ids, max_new_tokens=100)
print(tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

## Repository Structure

```
├── DocString_Fine_Tuning.ipynb   # Training notebook (data prep, LoRA config, SFTTrainer)
├── inference_demo.ipynb          # Load the fine-tuned model from HF Hub and run inference
└── README.md
```

## Key Learnings

- Set up an end-to-end LLM fine-tuning pipeline: dataset preparation → LoRA/SFT training → evaluation → pushing to Hugging Face Hub.
- Used `EarlyStoppingCallback` and validation loss tracking to monitor training.
- Learned to manage Colab runtime disconnects using checkpointing.
- Explored parameter-efficient fine-tuning (LoRA) to adapt a small LLM on limited compute (free-tier GPU).

## Future Improvements

- Train for more epochs with a larger/more diverse dataset sample for improved fluency and specificity.
- Add automated evaluation (BLEU/ROUGE against reference docstrings) instead of relying only on loss/accuracy.
- Deploy as a small web demo (Gradio Space) for interactive testing.

---

**Author:** Ubaid
