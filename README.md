# Emotion Classification with LoRA Fine-Tuning & Synthetic Data Generation

This project focuses on building an emotion classification model by combining synthetic data generation with parameter-efficient fine-tuning (LoRA).  
The workflow improves dataset diversity and trains a lightweight classifier with high accuracy across six emotion categories.

---

## Project Overview

The goal of the project is to classify text into six emotion labels:

- **sadness**
- **joy**
- **love**
- **anger**
- **fear**
- **surprise**

To enhance the dataset quality and class diversity, I generated synthetic sentences using an instruction-tuned LLM and merged them with the real *dair-ai/emotion* dataset.  
The model was then fine-tuned using **LoRA** to achieve strong performance with minimal additional parameters.

---

## Synthetic Data Generation

A 7B instruction-tuned model (**Mistral-7B-Instruct-v0.3**) was used to generate **100 high-quality sentences per emotion**.

Main steps:

1. **LLM generation with JSON-controlled prompts**
2. **Automatic extraction of the last valid JSON block**
3. **Normalization of samples (string → dict)**
4. **Deduplication across generations**
5. **Saving all 600 samples into `synthetic_emotions.json`**



---

## Dataset Preparation

### Real + Synthetic Merge

- Real dataset: **dair-ai/emotion**
- Synthetic dataset: 600 generated samples  
- Label mapping unified into numeric IDs (0–5)

### Preprocessing

- Train/validation split: **90/10**
- Tokenization with DistilBERT tokenizer  
- Truncation/padding to max length 128  
- Conversion to PyTorch tensors

---

## LoRA Fine-Tuning

LoRA was applied on top of **DistilBERT**, enabling parameter-efficient fine-tuning with minimal GPU usage.

**Training settings:**

- Batch size: 16  
- Learning rate: 2e-4  
- Epochs: 3  
- Weight decay: 0.01  
- Model saved every epoch  

Final model is stored in: /emotion_lora_model


---

## Evaluation Results

Evaluation was performed on the validation split using:

- **Accuracy**
- **Macro-F1**
- **Confusion Matrix**

| Metric | Score |
|--------|--------|
| Accuracy | **0.964** |
| Macro-F1 | **0.964** |

Confusion matrix analysis shows balanced prediction performance across all emotion categories.

---

## Technologies Used

- Python  
- HuggingFace Transformers  
- PEFT (LoRA)  
- Mistral-7B-Instruct  
- DistilBERT  
- HuggingFace Datasets  
- Pandas, NumPy  
- Matplotlib, Seaborn  

---

## Next Steps

Potential improvements:

- Test additional LLMs for synthetic generation  
- Explore QLoRA or other PEFT variants  
- Generate longer, more complex emotional sentences  
- Deploy the model as an API endpoint  

---
