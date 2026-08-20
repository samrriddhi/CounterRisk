# 🛡️ CounterRisk

An ML-driven fraud decision system that combines transaction-level machine learning, temporal transaction-network intelligence, and an adversarial Challenger to make fraud decisions more reliable and explainable.

**Defender → Challenger → Evidence → Final Decision**

---

## 📌 Features

### 🧠 Defender — ML Risk Model
- Random Forest fraud classification
- Transaction-level risk scoring
- Temporal train/test evaluation
- Precision, Recall, F1, ROC-AUC and PR-AUC evaluation

### 🥊 Challenger — Adversarial Review
- Independently reviews the Defender's assessment
- Detects potential overestimation or underestimation of risk
- Uses historical transaction-network evidence
- Explainable challenge reasoning

### 🕸️ Network Intelligence
- Transaction-wallet relationship analysis
- Historical wallet activity
- Prior connected transactions
- Prior confirmed illicit connections
- Historical illicit ratio
- Temporal leakage prevention

### ⚖️ Decision Engine
Produces:
- **ALLOW**
- **STEP-UP**
- **BLOCK**

### 🔍 Investigation Console
- Transaction investigation
- Defender assessment
- Challenger reasoning
- Network evidence
- Decision rationale

### 🧪 What-If Simulator
- Risk threshold simulation
- Network evidence threshold simulation
- Policy comparison

### 💳 Customer Payment Demo
Simulates how CounterRisk can evaluate payments behind a normal customer payment flow.

---

## 📊 Evaluation

### Defender

| Metric | Result |
|---|---:|
| Accuracy | 97.86% |
| Precision | 93.43% |
| Recall | 72.21% |
| F1 Score | 81.46% |
| ROC-AUC | 93.27% |
| PR-AUC | 80.02% |
| False Positives | 55 |
| False Negatives | 301 |

### CounterRisk

| Metric | Result |
|---|---:|
| Accuracy | **97.90%** |
| Precision | **93.79%** |
| Recall | **72.48%** |
| F1 Score | **81.77%** |
| False Positives | **52** |
| False Negatives | **298** |

**False positives reduced: 55 → 52**

---

## 🏗️ Architecture

```text
Transaction
     │
     ▼
  Defender
  ML Model
     │
     ▼
 Challenger
     │
     ├── Support
     └── Dispute
           │
           ▼
    Final Decision
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
  ALLOW  STEP-UP  BLOCK