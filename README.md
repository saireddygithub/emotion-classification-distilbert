# Fine-Tuning DistilBERT for Multi-Class Emotion Classification

A comparative study of a fine-tuned DistilBERT classifier against a tuned TF-IDF + Logistic Regression baseline on six-class English emotion classification.

**Research question:** To what extent does fine-tuning DistilBERT improve six-class emotion classification performance compared with a TF-IDF Logistic Regression baseline?

---

## Overview

The notebook implements the complete workflow end to end: public dataset acquisition, data auditing, leakage-aware preprocessing, a cross-validated classical baseline, DistilBERT tokenisation, a manual PyTorch fine-tuning loop, a controlled learning-rate experiment, a single final test evaluation, error and uncertainty analysis, and reproducible artefact saving.

**Key design decisions**

- **Macro F1 is the primary metric.** The dataset is class-imbalanced, so every emotion is given equal weight in model selection.
- **The test split is protected.** Test labels are never used for cleaning, tuning or checkpoint selection. The test set is evaluated exactly once, after both model configurations are frozen.
- **Leakage is removed and then asserted.** Duplicate and conflicting-label texts are dropped from training only; cross-split exact text overlap is removed and verified with runtime assertions.
- **The hyperparameter comparison is controlled.** Seeds, batch order, architecture and epoch budget are held fixed so learning rate is the only changed factor.

---

## Dataset

| Item | Value |
|---|---|
| Source | Kaggle — `praveengovi/emotions-dataset-for-nlp` |
| Files | `train.txt`, `val.txt`, `test.txt` (format: `sentence;emotion`) |
| Classes | sadness, joy, love, anger, fear, surprise |

Records after cleaning:

| Split | Records | Mean words | Median words | Max words |
|---|---|---|---|---|
| Train | 15,923 | 19.18 | 17 | 66 |
| Validation | 1,997 | 18.85 | 17 | 61 |
| Test | 2,000 | 19.15 | 17 | 61 |

The dataset is downloaded automatically at runtime through `kagglehub`, so no data files are stored in this repository. The dataset is public and normally requires no authentication; if Kaggle requests a token, add a Kaggle API token to your environment (in Colab, store it as a secret named `KAGGLE_API_TOKEN`).

---

## Method

### Baseline — TF-IDF + Logistic Regression
Unigram and bigram TF-IDF features with sublinear term frequency and `min_df=2`. Regularisation strength `C` and class weighting are selected by three-fold stratified cross-validation on the **training split only**, scored by macro F1.

### Transformer — DistilBERT
`distilbert/distilbert-base-uncased` with a dropout and linear classification head, fine-tuned end to end (66,958,086 trainable parameters).

```
Raw text → DistilBERT tokenizer → token IDs + attention mask → DistilBERT encoder
→ dropout + linear head → six-class logits → softmax probabilities → predicted label
```

| Setting | Value |
|---|---|
| Max sequence length | 128 (99th percentile is 55 tokens; ~0% truncation) |
| Train / eval batch size | 16 / 32 |
| Optimiser | AdamW, weight decay 0.01 |
| Schedule | 10% linear warm-up, then linear decay |
| Gradient clipping | Global norm 1.0 |
| Epochs | 3, with early stopping (patience 2) |
| Learning rates compared | 2e-5 and 3e-5 |
| Seed | 42 |

Training uses an explicit PyTorch loop: forward pass, cross-entropy loss, backpropagation, gradient clipping, optimiser and scheduler steps, per-epoch validation, and best-checkpoint saving on validation macro F1.

---

## Results

Learning-rate comparison (validation, best epoch):

| Learning rate | Best epoch | Validation macro F1 | Validation accuracy |
|---|---|---|---|
| 3e-5 (selected) | 3 | 0.9177 | 0.9409 |
| 2e-5 | 3 | 0.9123 | 0.9394 |

Final test-set performance (single evaluation, after both models were frozen):

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.8825 | 0.8317 | 0.8720 | 0.8487 | 0.8842 |
| Fine-tuned DistilBERT | **0.9340** | **0.9010** | **0.8904** | **0.8950** | **0.9338** |
| Difference | +0.0515 | +0.0692 | +0.0184 | +0.0463 | +0.0496 |

A paired bootstrap over 1,000 resamples of the test set gives a mean macro-F1 difference of 0.0461 with a 95% confidence interval of [0.022, 0.070], and the difference favoured DistilBERT in every resample.

**Error analysis highlights**

- Validation-to-test macro-F1 gap: 0.0228.
- Accuracy exceeds macro F1 by 0.0390, consistent with weaker performance on minority classes.
- `surprise` is both the smallest test class and the weakest class for DistilBERT.
- The most frequent directed confusion is joy → love (28 cases).
- 40 incorrect predictions carried confidence of at least 0.90, motivating future calibration work such as temperature scaling.

---

## Repository structure

```
emotion-classification-distilbert/
├── emotion_classification_distilbert.ipynb   # Complete end-to-end notebook
├── README.md
├── requirements.txt
└── LICENSE
```

Running the notebook creates an `emotion_assignment_outputs/` directory (and a matching `.zip`) containing the saved baseline pipeline, both DistilBERT checkpoints, all figures in PNG and PDF, per-class and aggregate metric tables, test predictions, error analysis, and `run_metadata.json` recording the configuration, dataset file hashes and package versions. These artefacts are generated at runtime and are not committed to the repository.

---

## Setup and usage

### Google Colab (recommended)
Open the notebook, select a GPU runtime (**Runtime → Change runtime type → T4 GPU**), and run all cells. The first cell installs the pinned dependencies. Total runtime is roughly 10–15 minutes on a T4.

### Local environment

```bash
git clone https://github.com/<your-username>/emotion-classification-distilbert.git
cd emotion-classification-distilbert

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook emotion_classification_distilbert.ipynb
```

Then run all cells in order. The notebook detects CUDA, Apple MPS or CPU automatically and adjusts its cache paths for Colab, Kaggle or local Jupyter. A GPU is strongly recommended; CPU-only fine-tuning is considerably slower.

**PyTorch note:** install the build matching your CUDA driver rather than the generic wheel, for example `pip install torch --index-url https://download.pytorch.org/whl/cu128`. The reference run used torch 2.10.0+cu128 on a Tesla T4.

### Inference on new text

After running the notebook, the `predict_emotion` helper (Phase 14) returns the predicted emotion and its confidence for any string or list of strings:

```python
predict_emotion("i feel completely overwhelmed by everything today",
                best_transformer, tokenizer)
```

---

## Reproducibility

- All random seeds (Python, NumPy, PyTorch, CUDA) are fixed at 42; deterministic cuDNN kernels are enabled.
- A freshly seeded DataLoader is constructed for each learning-rate experiment so both runs receive an identical batch order.
- SHA-256 hashes of the downloaded Kaggle files are recorded in `run_metadata.json`.
- Library versions are pinned; the exact runtime environment is printed in Phase 1 and saved with the run metadata.

Reference environment: Python 3.12.13, PyTorch 2.10.0+cu128, Transformers 5.13.1, Datasets 5.0.0, scikit-learn 1.6.1, Tesla T4 GPU.

---

## Limitations

- Results come from one English emotion dataset and may not generalise to other domains or languages.
- Only one BERT-style architecture and two learning rates were compared, within the available compute budget.
- A single seed was used; repeated-seed training would better estimate optimisation variability.
- Minority classes have few examples, so accuracy can conceal weaker class-balanced performance.
- Some errors likely reflect short, ambiguous or inconsistently annotated text rather than model failure alone.

---

## Licence

Released under the MIT Licence. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Dataset: `praveengovi/emotions-dataset-for-nlp` on Kaggle.
- Pretrained model: `distilbert/distilbert-base-uncased` (Hugging Face).
- Built with PyTorch, Hugging Face Transformers and Datasets, scikit-learn, pandas and matplotlib.
