# 🔍 HDFS Log Anomaly Detection with Deep Learning (DeepLog-style LSTM)

> **L2 Deep Learning Mini-Project** — Intrusion & Anomaly Detection from HDFS Logs using NLP-style Sequence Modeling.

---

## 📋 Overview

This project implements a complete end-to-end prototype for **log anomaly detection** on the **HDFS_v1** dataset using a **DeepLog-inspired LSTM** neural network.

The pipeline covers:

```
HDFS raw logs → Preprocessing & Parsing → Log Templates / Tokenization
    → Sequence Creation → LSTM Classifier → Anomaly / Normal Prediction
        → Evaluation (Precision, Recall, F1) → Explainable AI (XAI)
```

Each HDFS block (identified by a `blk_...` ID) is treated as a **sequence of log-template tokens** and classified as either **Normal** or **Anomaly**.

---

## 📂 Dataset

The notebook is designed for the **HDFS_v1** dataset, which contains:

| File | Description |
|---|---|
| `HDFS.log` | Raw HDFS log file (~11 million lines) |
| `anomaly_label.csv` | Ground-truth labels per block ID (`Normal` / `Anomaly`) |

> ⚠️ **Before running on Kaggle**, add the HDFS_v1 dataset as input via **Add Input**.  
> The notebook automatically searches `/kaggle/input` for the required files.

---

## 🏗️ Pipeline — Step by Step

| Step | Description |
|---|---|
| **1. Libraries** | Imports NumPy, Pandas, PyTorch, scikit-learn; sets random seeds for reproducibility |
| **2. Data Discovery** | Recursively searches Kaggle input for `HDFS.log` and `anomaly_label.csv` |
| **3. Load Labels** | Reads `anomaly_label.csv`, normalises column names, maps `Normal → 0` / `Anomaly → 1` |
| **4. Parse Logs** | Regex-based parser extracts `Date, Time, Pid, Level, Component, Content`; groups events by `BlockId` |
| **5. Templatisation** | Replaces dynamic values (`blk_*`, IPs, numbers, paths) with `<BLK>`, `<IP>`, `<NUM>`, `<PATH>`; maps templates to integer IDs |
| **6. Sequences** | Groups ordered template-ID lists per block; adds block-level label |
| **7. Padding** | Pads/truncates sequences to `MAX_LEN` (95th percentile of sequence lengths, capped at 100) |
| **8. Split** | Stratified 70 / 15 / 15 train / val / test split |
| **9. DataLoaders** | PyTorch `Dataset` + `DataLoader` with batch size 128 |
| **10. LSTM Model** | Embedding → LSTM → Dropout → Linear (`LogLSTMClassifier`) |
| **11. Training Setup** | Weighted `CrossEntropyLoss` (handles class imbalance) + Adam optimiser |
| **12. Training** | 5 epochs with best-model checkpoint saved to `best_hdfs_lstm_model.pt` |
| **13. Evaluation** | Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC/AUC on the test set |
| **14. Predictions** | Sample prediction table with anomaly probabilities |
| **15. XAI** | Gradient × Embedding saliency + Occlusion/Perturbation + Global template importance |
| **16. Save Outputs** | Exports model, vocabulary, predictions, metrics, and XAI results |

---

## 🧠 Model Architecture

```
Input: Sequence of log-template IDs  [batch, MAX_LEN]
         ↓
Embedding Layer (vocab_size × 64, padding_idx=0)
         ↓
LSTM (hidden=128, layers=1)
         ↓
Last Hidden State  [batch, 128]
         ↓
Dropout (p=0.3)
         ↓
Linear (128 → 2)   →  Normal / Anomaly
```

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Embedding dim | 64 |
| LSTM hidden dim | 128 |
| LSTM layers | 1 |
| Dropout | 0.3 |
| Batch size | 128 |
| Epochs | 5 |
| Optimiser | Adam (lr=1e-3) |
| Loss | Weighted CrossEntropy |
| Random seed | 42 |

---

## 📊 Evaluation Metrics

After training, the model is evaluated on the held-out test set using:

- **Accuracy** — overall proportion of correct predictions
- **Precision** — of all predicted anomalies, how many are truly anomalous
- **Recall** — of all true anomalies, how many were detected *(critical for security)*
- **F1-score** — harmonic mean of Precision and Recall
- **Confusion Matrix** — visual breakdown of TP, TN, FP, FN
- **ROC / AUC** — trade-off between TPR and FPR at different thresholds

> 💡 For anomaly detection tasks, **Recall and F1-score are the most important metrics** since missing a real anomaly (false negative) is more costly than a false alarm.

---

## 🔎 Explainable AI (XAI)

Three interpretability methods are implemented to explain *why* the model flags a sequence as anomalous:

### 15.1 Gradient × Embedding (Local)
Backpropagates the anomaly score through the embedding layer to measure each token's influence on the prediction.

### 15.2 Occlusion / Perturbation (Local, Model-Agnostic)
Replaces one log event at a time with padding (`0`) and measures how much the anomaly probability drops — the larger the drop, the more important the event.

### 15.3 Global Template Importance
Aggregates Gradient × Embedding scores across up to 200 test samples to reveal which log templates the model **globally** relies on most for anomaly detection.

---

## 📁 Output Files

Running the full notebook generates the following files:

| File | Description |
|---|---|
| `best_hdfs_lstm_model.pt` | Best LSTM model weights (saved by validation F1) |
| `template_to_id.json` | Log template → integer ID vocabulary |
| `hdfs_lstm_test_predictions.csv` | Per-block test predictions with probabilities |
| `test_metrics.csv` | Summary table of Accuracy, Precision, Recall, F1 |
| `classification_report.csv` | Detailed per-class classification report |
| `confusion_matrix.csv` | Confusion matrix as a table |
| `confusion_matrix.png` | Confusion matrix figure (300 dpi) |
| `evaluation_metrics_bar_chart.png` | Bar chart of evaluation metrics (300 dpi) |
| `roc_curve.png` | ROC curve with AUC score (if computable) |
| `xai_gradient_example.csv` | Gradient × Embedding explanation for one example |
| `xai_occlusion_example.csv` | Occlusion explanation for one example |
| `xai_global_template_importance.csv` | Aggregated global template importance |

---

## ⚙️ Requirements

```
python >= 3.10
torch >= 2.0
numpy
pandas
scikit-learn
matplotlib
```

Install with:
```bash
pip install torch numpy pandas scikit-learn matplotlib
```

> The notebook was developed and tested on **Kaggle** with GPU acceleration (P100/T4).  
> Set the accelerator to **GPU** in Kaggle notebook settings for faster training.

---

## 🚀 How to Run

1. **Upload to Kaggle** and open `mini-project-dl-v2.ipynb`.
2. Add the **HDFS_v1 dataset** as input (via *Add Input* → search `HDFS_v1`).
3. Set the accelerator to **GPU** (Settings → Accelerator → GPU T4 x2 or P100).
4. Click **Run All** (`Kernel → Restart & Run All`).
5. Outputs will be saved in `/kaggle/working/`.

---

## 📌 Project Context

This notebook was developed as part of the **L2 Deep Learning course** at **SDIA**.  
The goal is to study **intrusion and anomaly detection** from system logs using modern Deep Learning and NLP-style sequence modelling techniques.

The approach is inspired by the seminal [**DeepLog**](https://dl.acm.org/doi/10.1145/3133956.3134015) paper (Du et al., CCS 2017), which pioneered LSTM-based log anomaly detection.

---

## 📖 References

- Du, M., Li, F., Zheng, G., & Srikumar, V. (2017). **DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning**. *ACM CCS 2017*.
- He, P., Zhu, J., Zheng, Z., & Lyu, M. R. (2017). **Drain: An Online Log Parsing Approach with Fixed Depth Tree**. *IEEE ICWS 2017*.
- HDFS_v1 dataset: [Loghub — HDFS](https://github.com/logpai/loghub)
