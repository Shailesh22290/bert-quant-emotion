# BERT Emotion Classification with Quantization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Implementation of BERT quantization techniques for emotion classification on the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset. This project explores model compression through Post-Training Quantization (PTQ), Quantization-Aware Training (QAT), and Quantized Low-Rank Adaptation (QLoRA).

## 🎯 Project Overview

This repository contains implementations of various BERT compression techniques to reduce model size and improve inference speed while maintaining accuracy on emotion classification tasks.

**Task**: Multi-class emotion classification (6 classes: sadness, joy, love, anger, fear, surprise)

**Dataset**: [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
- Train: 16,000 samples
- Validation: 2,000 samples  
- Test: 2,000 samples

## 📊 Results Summary

> **Status**: 🚧 Work in Progress - Will be updated as models are trained

| Model Variant | Macro F1 | Accuracy | Size (MB) | Latency (ms) | Status |
|---------------|----------|----------|-----------|--------------|---------|
| Baseline (FP32) | TBD | TBD | TBD | TBD | ⏳ Pending |
| PTQ (INT8) | TBD | TBD | TBD | TBD | ⏳ Pending |
| QAT (INT8) | TBD | TBD | TBD | TBD | ⏳ Pending |
| QLoRA | TBD | TBD | TBD | TBD | ⏳ Pending |

## 🗂️ Repository Structure
```
bert-emotion-quantization/
│
├── weights/                          # Model weights (uploaded as trained)
│   ├── baseline/                     # ⏳ Coming soon
│   ├── ptq/                          # ⏳ Coming soon
│   ├── qat/                          # ⏳ Coming soon
│   └── qlora/                        # ⏳ Coming soon
│
├── configs/                          # Configuration files
│   └── training_config.json          # ⏳ Coming soon
│
├── notebooks/                        # Colab notebooks
│   └── inference_evaluation.ipynb    # ⏳ Coming soon
│
├── results/                          # Evaluation results
│   ├── confusion_matrices/           # ⏳ Coming soon
│   └── metrics/                      # ⏳ Coming soon
│
├── docs/                             # Documentation
│   └── report.pdf                    # ⏳ Coming soon
│
├── README.md                         # This file
└── requirements.txt                  # ⏳ Coming soon
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (recommended)
Google Colab account (for notebook execution)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/shailesh22290/bert-emotion-quantization.git
cd bert-emotion-quantization

# Install dependencies (once requirements.txt is added)
pip install -r requirements.txt
```

### Running Inference

> **Note**: Colab notebook will be available soon. It will automatically download weights from this repository.
```bash
# Instructions will be added once the notebook is ready
# Expected runtime: ≤15 minutes per model variant on Colab GPU
```

## 📝 Implementation Details

### 1️⃣ Baseline Model (FP32/FP16)
**Status**: ⏳ In Progress

- **Model**: `bert-base-uncased`
- **Training**:
  - Epochs: TBD
  - Learning rate: TBD
  - Batch size: TBD
  - Max sequence length: TBD
- **Performance**: TBD

### 2️⃣ Post-Training Quantization (PTQ)
**Status**: ⏳ Pending

- **Quantization method**: TBD (Dynamic/Static)
- **Bit-width**: INT8
- **Layers quantized**: TBD
- **Calibration**: TBD
- **Performance**: TBD

### 3️⃣ Quantization-Aware Training (QAT)
**Status**: ⏳ Pending

- **QAT configuration**: TBD
- **Frozen layers**: TBD
- **Training strategy**: TBD
- **Performance**: TBD

### 4️⃣ Quantized LoRA (QLoRA)
**Status**: ⏳ Pending

- **Base quantization**: TBD
- **LoRA rank (r)**: TBD
- **Alpha**: TBD
- **Target modules**: TBD
- **Performance**: TBD

## 📈 Evaluation Metrics

All models are evaluated on:
- ✅ Macro F1-score (primary metric)
- ✅ Accuracy
- ✅ Per-class F1 scores
- ✅ Confusion matrix
- ✅ Model size (MB)
- ✅ Inference latency (ms/example)

## 🔄 Project Timeline

### Phase 1: Setup & Baseline ✅ Current Phase
- [x] Repository setup
- [x] README creation
- [ ] Dataset exploration
- [ ] Baseline model training

### Phase 2: PTQ Implementation ⏳ Upcoming
- [ ] PTQ implementation
- [ ] Calibration & evaluation
- [ ] Weight upload

### Phase 3: QAT Implementation ⏳ Upcoming
- [ ] QAT setup
- [ ] Training with quantization simulation
- [ ] Evaluation & comparison

### Phase 4: QLoRA Implementation ⏳ Upcoming
- [ ] QLoRA configuration
- [ ] Adapter training
- [ ] Final evaluation

### Phase 5: Analysis & Documentation ⏳ Upcoming
- [ ] Comprehensive comparison
- [ ] Report writing
- [ ] Notebook finalization

**Expected Completion**: November 20, 2025

## 🛠️ Technologies Used

- **Framework**: PyTorch, Hugging Face Transformers
- **Quantization**: `torch.quantization`, `torch.ao.quantization`
- **LoRA**: PEFT library, LLaMA Factory
- **Dataset**: Hugging Face Datasets
- **Evaluation**: scikit-learn, seaborn

## 📚 References

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
- [dair-ai/emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion)



## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Course Instructor: Dr. Jasabanta Patro
- Hugging Face for datasets and model hosting
- PyTorch team for quantization tools

---

**Last Updated**: 22nd October 2025  
**Status**: 🚧 Active Development

> ⭐ Star this repository if you find it helpful!
