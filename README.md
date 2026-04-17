# 🤖 BART Uzbek Text Summarization — Fine-Tuning

<p align="center">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="80"/>
  &nbsp;&nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="70"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-BART--large--CNN-blue?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/Dataset-XLSum%20Uzbek-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Framework-HuggingFace-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/GPU-T4-red?style=for-the-badge&logo=nvidia"/>
</p>

---

## 📌 Loyiha haqida

Bu loyihada **Facebook BART-large-CNN** modeli o'zbek tilida matn qisqartirish (summarization) uchun **fine-tune** qilingan.

Dataset sifatida **BBC XLSum v2.0** ning o'zbek tili qismi ishlatilgan.

---

## 📂 Dataset

| Split | Miqdor |
|-------|--------|
| 🟢 Train | 4,460 ta maqola |
| 🟡 Validation | 635 ta maqola |
| 🔵 Test | 634 ta maqola |

**Manba:** [XLSum — Multilingual Summarization Dataset](https://huggingface.co/datasets/csebuetnlp/xlsum)

---

## 🏗️ Model Arxitekturasi

```
facebook/bart-large-cnn
├── Encoder (12 layers)
├── Decoder (12 layers)
└── Language Model Head
```

- **Max input length:** 128 tokens
- **Max output length:** 64 tokens
- **Precision:** FP16 (half-precision)

---

## ⚙️ Training Konfiguratsiyasi

| Parametr | Qiymat |
|----------|--------|
| 🔁 Epochs | 3 |
| 📦 Batch size | 4 per device |
| 📈 Gradient accumulation | 4 steps (effektiv batch = 16) |
| ⚡ FP16 | ✅ Ha |
| 🛑 Early stopping | patience = 2 |
| 📉 Weight decay | 0.01 |
| 🔥 Warmup steps | 200 |
| 💾 GPU | NVIDIA T4 |

---

## 📊 Training Natijalari

| Epoch | Training Loss | Validation Loss | Rouge1 | Rouge2 | RougeL |
|-------|--------------|-----------------|--------|--------|--------|
| 1 | 5.9343 | 1.3623 | 0.0010 | 0.0005 | 0.0010 |
| 2 | 4.8068 | 1.1554 | 0.0023 | 0.0004 | 0.0023 |
| 3 | 4.1422 | 1.1064 | 0.0075 | 0.0020 | 0.0075 |

### 🧪 Test Natijalari

| Metrika | Qiymat |
|---------|--------|
| eval_loss | 1.1486 |
| Rouge1 | 0.0070 |
| Rouge2 | 0.0017 |
| RougeL | 0.0069 |

> ⚠️ **Natijalar past sababi:** BART asosan ingliz tili uchun pre-trained. O'zbek tili uchun `google/mt5-base` yoki multilingual model ishlatish tavsiya etiladi.

---

## 🚀 Qanday ishlatish

### 1. Reponi klonlash

```bash
git clone https://github.com/username/repo-nomi.git
cd repo-nomi
```

### 2. Kerakli kutubxonalar o'rnatish

```bash
pip install transformers datasets evaluate rouge_score
```

### 3. Google Colab da ochish

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 🔮 Keyingi Qadamlar

- [ ] 🌍 `google/mt5-base` modeliga o'tish
- [ ] 📈 Ko'proq epoch bilan train qilish
- [ ] 🧹 Dataset preprocessing yaxshilash
- [ ] 🚀 Hugging Face Hub ga yuklash

---

## 🛠️ Texnologiyalar

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)
![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab)

---

## 👨‍💻 Muallif

**Jonibek Abdurahmonov**

---

<p align="center">⭐ Agar foydali bo'lsa, yulduzcha bosing!</p>
