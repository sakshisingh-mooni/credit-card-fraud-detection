# Credit Card Fraud Detection

> XGBoost fraud classifier trained on 284,807 real transactions · deployed as a REST API on Azure · PR-AUC 0.87 · ROC-AUC 0.97

**Live API** → `https://fraud-detection-api-sakshi.azurewebsites.net`

---

## What this project does

Detects fraudulent credit card transactions in real time. Given 30 transaction features (28 PCA-anonymised + Time + Amount), the model returns a fraud probability and a binary label in under 100 ms.

The dataset is the [MLG-ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 European transactions from 2013, of which only 492 (0.17%) are fraud. The extreme class imbalance is the core technical challenge.

---

## Architecture

```
Client (curl / Postman / app)
        │
        │ HTTPS
        ▼
┌─────────────────────────────────────────┐
│   Azure Web App for Containers (B1)     │
│   Central India · fraud-detection-api   │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  Gunicorn (2 workers × 2 threads)│   │
│  │  ┌────────────────────────────┐  │   │
│  │  │  Flask 3.x REST API        │  │   │
│  │  │  ├── GET  /health          │  │   │
│  │  │  ├── GET  /info            │  │   │
│  │  │  ├── POST /predict         │  │   │
│  │  │  └── POST /predict/batch   │  │   │
│  │  └──────────────┬─────────────┘  │   │
│  └─────────────────│────────────────┘   │
│                    │ joblib.load        │
│  ┌─────────────────▼────────────────┐   │
│  │  model/                          │   │
│  │  ├── fraud_model.pkl  (XGBoost)  │   │
│  │  ├── scaler.pkl  (StandardScaler)│   │
│  │  ├── feature_cols.pkl            │   │
│  │  └── metadata.json               │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
        ▲
        │ docker pull
        │
┌───────────────────────────┐
│  Azure Container Registry │
│  frauddetectionacrmooni   │
│  fraud-api:v1             │
└───────────────────────────┘
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Model | XGBoost 3.2 · scikit-learn 1.9 |
| Imbalance handling | `scale_pos_weight=577` · SMOTE (CV only) |
| Explainability | SHAP TreeExplainer |
| API | Flask 3.1 · Gunicorn 26 |
| Containerisation | Docker (python:3.12-slim) |
| Registry | Azure Container Registry (Basic) |
| Hosting | Azure Web App for Containers (B1 Linux) |
| Testing | pytest · 21 tests · Flask test client |

---

## Model performance

Evaluated on a held-out 20% test split (56,962 transactions, 98 fraud cases).

| Metric | Score |
|---|---|
| **PR-AUC** | **0.87** |
| ROC-AUC | 0.97 |
| Fraud recall | 79% |
| Fraud precision | 95% |
| Optimal threshold | 0.99 |

**Why PR-AUC over ROC-AUC?**  
With 0.17% fraud, a model that predicts "legitimate" for every transaction scores 99.8% accuracy and ROC-AUC ~0.5. PR-AUC is the right metric for severe class imbalance — it measures performance specifically on the positive (fraud) class.

**Why threshold = 0.99?**  
`scale_pos_weight=577` tells XGBoost that missing a fraud costs 577× more than a false alarm. The model pushes fraud probabilities very high for suspicious transactions (often > 0.995). The threshold was tuned by maximising F1 on the precision-recall curve — 0.99 is where precision and recall trade off optimally for this model's output distribution, not a naive choice.

---

## Key ML decisions

**Split before scaling** — `StandardScaler` is fit only on training data, then applied to the test set. Fitting on the full dataset before splitting is data leakage.

**SMOTE inside CV folds only** — synthetic oversampling is applied within each training fold during cross-validation. Applying SMOTE before splitting introduces synthetic fraud samples into the validation set, inflating recall.

**`scale_pos_weight` on the final model** — XGBoost's built-in class weighting handles imbalance during full training without requiring SMOTE, keeping the feature distribution clean.

**SHAP for explainability** — `TreeExplainer` gives exact (not approximate) Shapley values for tree models. Top fraud predictors: V14, V17, V12, V10, V4.

---

## API reference

**Base URL:** `https://fraud-detection-api-sakshi.azurewebsites.net`

### GET /health
Liveness probe. Returns 200 when the model is loaded and the server is ready.

```bash
curl https://fraud-detection-api-sakshi.azurewebsites.net/health
```
```json
{"model_version": "1.0.0", "status": "ok"}
```

### GET /info
Model metadata — version, threshold, metrics, training parameters.

```bash
curl https://fraud-detection-api-sakshi.azurewebsites.net/info
```
```json
{
  "model_version": "1.0.0",
  "optimal_threshold": 0.99,
  "feature_count": 30,
  "test_pr_auc": 0.87,
  "test_roc_auc": 0.97,
  "scale_pos_weight": 577.29,
  "training_fraud_rate": 0.00173
}
```

### POST /predict
Single transaction prediction.

**Request body** — 30 floats in order: V1–V28 (PCA), Time (seconds), Amount (USD).

```bash
curl -X POST https://fraud-detection-api-sakshi.azurewebsites.net/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'
```
```json
{
  "fraud_probability": 0.000355,
  "prediction": 0,
  "label": "LEGITIMATE",
  "threshold_used": 0.99,
  "model_version": "1.0.0"
}
```

**Validation errors return 400** with a descriptive message — wrong feature count, non-numeric values, missing key, invalid JSON.

### POST /predict/batch
Up to 1,000 transactions per request. Partial errors are collected per-index without failing the entire batch.

```bash
curl -X POST https://fraud-detection-api-sakshi.azurewebsites.net/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"features": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},
      {"features": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
    ]
  }'
```
```json
{
  "results": [
    {"index": 0, "fraud_probability": 0.000355, "prediction": 0, "label": "LEGITIMATE"},
    {"index": 1, "fraud_probability": 0.000355, "prediction": 0, "label": "LEGITIMATE"}
  ],
  "total": 2,
  "processed": 2,
  "fraud_count": 0,
  "error_count": 0
}
```

---

## Run locally

**Prerequisites:** Python 3.12+, Docker Desktop

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection
cd credit-card-fraud-detection

# Install API dependencies
pip install -r requirements.txt

# Run tests (requires model artifacts in tests/fixtures/)
python -m pytest tests/ -v
# Expected: 21 passed

# Start the API
python app.py
# or with gunicorn (same as Azure):
gunicorn --bind 0.0.0.0:8000 --workers 2 --threads 2 app:app

# Test it
curl http://localhost:8000/health
```

**To retrain the model**, open `credit_card_fraud_detection_v2.ipynb`, add `creditcard.csv` to the project root, and run Kernel → Restart & Run All. The last cell exports all artifacts to `model/`.

Install notebook dependencies:
```bash
pip install -r requirements-notebook.txt
```

---

## Project structure

```
credit-card-fraud-detection/
├── app.py                          ← Flask REST API
├── Dockerfile                      ← python:3.12-slim, non-root user
├── .dockerignore
├── requirements.txt                ← API runtime deps only
├── requirements-notebook.txt       ← training deps (pandas, shap, imbalanced-learn)
├── SAVE_MODEL_CELL.py              ← paste as last notebook cell to export model
├── credit_card_fraud_detection_v2.ipynb
├── model/                          ← serialised artifacts (gitignored)
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   ├── feature_cols.pkl
│   └── metadata.json
└── tests/
    ├── fixtures/                   ← minimal test artifacts (not the trained model)
    ├── conftest.py                 ← sets MODEL_DIR before app import
    └── test_api.py                 ← 21 smoke tests
```

---

## Deployment (Azure)

The full step-by-step is in [`AZURE_DEPLOYMENT_GUIDE.md`](AZURE_DEPLOYMENT_GUIDE.md).

**Summary:**
```bash
# Build and push to Azure Container Registry
az acr build --registry frauddetectionacrmooni --image fraud-api:v1 .

# Deploy to Azure Web App
az webapp create \
  --resource-group fraud-detection-rg \
  --plan fraud-plan \
  --name fraud-detection-api-sakshi \
  --deployment-container-image-name frauddetectionacrmooni.azurecr.io/fraud-api:v1
```

**Cost:** Azure for Students subscription · ~₹0 within free credit limits.

---

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — MLG-ULB, Université Libre de Bruxelles.  
284,807 transactions · 492 fraud (0.17%) · 28 PCA-anonymised features + Time + Amount.  
The CSV is not committed to this repository.

---

## License

MIT
