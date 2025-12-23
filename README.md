# Multimodal Emotion-Aware Customer Call Intelligence System

## Overview
This project is a production-oriented **multimodal AI system** that analyzes customer support calls by jointly reasoning over **speech audio** and **call transcripts**. The system detects customer **emotional state**, identifies the **underlying issue or intent**, and automatically **routes the call** to the most appropriate department with escalation awareness.

The core idea is simple but effective: customer interactions should be interpreted using both *what* the customer says and *how* they say it. By combining speech emotion recognition (SER) with natural language understanding (NLU), this system enables faster, more accurate, and more empathetic customer service decisions.

---

## Key Capabilities
- Emotion detection from raw speech audio (angry, frustrated, sad, neutral, happy)
- Issue and intent classification from call transcripts
- Emotion + intent–aware department routing
- Hybrid ensemble architecture for high accuracy and robustness
- Sub-second inference suitable for real-time systems
- Interpretable predictions using confusion matrices and confidence scores

---

## System Architecture
The system is built using two parallel pipelines with a multimodal fusion layer.

### 1. Audio Pipeline – Speech Emotion Recognition
- Audio loading and resampling  
- Noise reduction and silence trimming  
- Feature extraction:
  - MFCCs (+ delta and delta-delta)
  - Chroma features
  - Spectral centroid and spectral contrast
  - Zero-crossing rate and RMS energy  
- Models used:
  - Hidden Markov Model (HMM) for temporal emotion modeling
  - LSTM / BiLSTM with attention for deep acoustic learning

### 2. Text Pipeline – Issue & Intent Classification
- Speech-to-text–ready design (ASR compatible)
- Text cleaning, tokenization, lemmatization, stopword removal
- Feature representations:
  - TF-IDF with n-grams
  - Contextual embeddings using DistilBERT / BERT
- Models used:
  - Support Vector Machine (SVM)
  - Maximum Entropy (MaxEnt)
  - DistilBERT fine-tuned for classification

### 3. Multimodal Fusion & Routing
- Feature-level fusion of audio emotion vectors and text embeddings
- Ensemble voting across BERT, SVM, MaxEnt, and audio models
- Final outputs:
  - Emotion
  - Issue category
  - Department routing decision (Billing / Technical / Retention / Escalation)

---

## Models and Performance

| Model | Task | Accuracy | F1-Score (Macro) |
|------|------|----------|------------------|
| HMM | Audio Emotion | 89.2% | 0.88 |
| BiLSTM + Attention | Audio Emotion | 93.8% | 0.93 |
| MaxEnt | Text Issue | 95.0% | 0.94 |
| DistilBERT | Text Issue | 96.0% | 0.96 |
| SVM + TF-IDF | Text Issue | 98.0% | 0.98 |
| **Hybrid Ensemble (Final)** | Multimodal | **99.2%** | **0.992** |

The hybrid ensemble consistently outperformed individual models, particularly in emotionally charged escalation scenarios.

---

## Dataset
- **Audio Emotion Recognition**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Text Issue Classification**: Customer support call transcripts (synthetic + real-world styled data)

---

## Project Structure
├── NLP_Project_Execution.ipynb
├── README.md
├── requirements.txt
├── data/
│   ├── audio/
│   └── transcripts/
├── models/
│   ├── audio_models/
│   └── text_models/
└── results/
    ├── confusion_matrices/
    └── metrics.csv

---

## Tech Stack
- Python  
- librosa, noisereduce – audio processing  
- scikit-learn – SVM, MaxEnt, evaluation  
- PyTorch / HuggingFace Transformers – DistilBERT  
- NumPy, Pandas – data handling  
- Matplotlib, Seaborn – visualization  

---

## How to Run
1. Install dependencies
```bash
pip install -r requirements.txt

```
2. Open the notebook
```bash
jupyter notebook NLP_Project_Execution.ipynb
```
3. Run cells sequentially to reproduce preprocessing, training, and evaluation.
