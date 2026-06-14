"""
app.py — Credit Card Fraud Detection REST API
Flask 3.x  |  Gunicorn  |  Azure Web App / Hugging Face Spaces

Routes:
  GET  /health          → liveness probe (Azure & load balancer use this)
  GET  /info            → model metadata (version, threshold, metrics)
  POST /predict         → single transaction prediction
  POST /predict/batch   → batch prediction (list of transactions)

Model artifact:
  model/fraud_pipeline.joblib — sklearn Pipeline (ColumnTransformer + XGBClassifier)
  Preprocessing (StandardScaler on Time/Amount) and inference are a single unit —
  no separate scaler file, no manual transform step, no train/serve skew risk.

Sources:
  Flask 3.x docs:   https://flask.palletsprojects.com/en/3.1.x/
  sklearn Pipeline: https://scikit-learn.org/stable/modules/pipeline.html
  joblib:           https://joblib.readthedocs.io/en/stable/persistence.html
  Azure Web App:    https://learn.microsoft.com/en-us/azure/app-service/
"""

import os
import json
import logging
import numpy as np
import joblib
from flask import Flask, request, jsonify

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Load artifacts at startup ─────────────────────────────────────────────────
MODEL_DIR = os.environ.get('MODEL_DIR', 'model')

try:
    # Single pipeline artifact: ColumnTransformer(scaler) + XGBClassifier
    # No separate scaler.pkl or feature_cols.pkl needed at inference time.
    pipeline = joblib.load(os.path.join(MODEL_DIR, 'fraud_pipeline.joblib'))

    with open(os.path.join(MODEL_DIR, 'metadata.json')) as f:
        metadata = json.load(f)

    THRESHOLD     = metadata['optimal_threshold']
    FEATURE_COUNT = metadata['feature_count']      # 30
    FEATURE_ORDER = metadata['feature_order']      # [V1..V28, Time, Amount]

    logger.info('Pipeline artifact loaded successfully.')
    logger.info(f'Model version : {metadata["model_version"]}')
    logger.info(f'Threshold     : {THRESHOLD}')
    logger.info(f'Test PR-AUC   : {metadata["test_pr_auc"]}')

except Exception as e:
    logger.error(f'FATAL: Could not load model artifacts from {MODEL_DIR}: {e}')
    raise


# ── Helper ────────────────────────────────────────────────────────────────────
def make_prediction(features: list) -> dict:
    """
    Run inference through the sklearn Pipeline.

    The Pipeline's ColumnTransformer handles StandardScaler on Time/Amount
    internally — no manual preprocessing needed here.

    Parameters
    ----------
    features : list of 30 floats in order [V1..V28, Time, Amount]

    Returns
    -------
    dict with fraud_probability, prediction, label, threshold_used, model_version
    """
    X = np.array(features, dtype=np.float64).reshape(1, -1)
    prob  = float(pipeline.predict_proba(X)[0, 1])
    pred  = int(prob >= THRESHOLD)
    label = 'FRAUD' if pred == 1 else 'LEGITIMATE'

    return {
        'fraud_probability': round(prob, 6),
        'prediction':        pred,
        'label':             label,
        'threshold_used':    THRESHOLD,
        'model_version':     metadata['model_version']
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get('/health')
def health():
    """
    Liveness probe.
    Azure App Service and load balancers call this to verify the container is up.
    Source: https://learn.microsoft.com/en-us/azure/app-service/monitor-instances-health-check
    """
    return jsonify({'status': 'ok', 'model_version': metadata['model_version']}), 200


@app.get('/info')
def info():
    """Return model metadata — useful for monitoring and audit trails."""
    return jsonify(metadata), 200


@app.post('/predict')
def predict():
    """
    Single transaction prediction.

    Expected JSON body:
    {
      "features": [V1, V2, ..., V28, Time, Amount]   // 30 floats, in this exact order
    }

    Response:
    {
      "fraud_probability": 0.042,
      "prediction": 0,
      "label": "LEGITIMATE",
      "threshold_used": 0.99,
      "model_version": "2.0.0"
    }
    """
    data = request.get_json(force=True, silent=True)

    if data is None:
        return jsonify({'error': 'Request body must be valid JSON'}), 400

    if 'features' not in data:
        return jsonify({'error': "Missing required key: 'features'"}), 400

    features = data['features']

    if not isinstance(features, list):
        return jsonify({'error': "'features' must be a JSON array"}), 400

    if len(features) != FEATURE_COUNT:
        return jsonify({
            'error': f'Expected {FEATURE_COUNT} features, received {len(features)}'
        }), 400

    try:
        features = [float(v) for v in features]
    except (TypeError, ValueError) as e:
        return jsonify({'error': f'All feature values must be numeric: {e}'}), 400

    try:
        result = make_prediction(features)
        logger.info(
            f'/predict | prob={result["fraud_probability"]:.4f} | label={result["label"]}'
        )
        return jsonify(result), 200

    except Exception as e:
        logger.error(f'/predict error: {e}')
        return jsonify({'error': 'Prediction failed. Check server logs.'}), 500


@app.post('/predict/batch')
def predict_batch():
    """
    Batch prediction — up to 1000 transactions per request.

    Expected JSON body:
    {
      "transactions": [
        {"features": [V1..V28, Time, Amount]},
        {"features": [V1..V28, Time, Amount]},
        ...
      ]
    }
    """
    BATCH_LIMIT = 1000

    data = request.get_json(force=True, silent=True)

    if data is None:
        return jsonify({'error': 'Request body must be valid JSON'}), 400

    if 'transactions' not in data:
        return jsonify({'error': "Missing required key: 'transactions'"}), 400

    transactions = data['transactions']

    if not isinstance(transactions, list) or len(transactions) == 0:
        return jsonify({'error': "'transactions' must be a non-empty array"}), 400

    if len(transactions) > BATCH_LIMIT:
        return jsonify({'error': f'Batch limit is {BATCH_LIMIT} transactions'}), 400

    results = []
    errors  = []

    for i, txn in enumerate(transactions):
        if 'features' not in txn:
            errors.append({'index': i, 'error': "Missing 'features' key"})
            continue

        features = txn['features']
        if len(features) != FEATURE_COUNT:
            errors.append({
                'index': i,
                'error': f'Expected {FEATURE_COUNT} features, got {len(features)}'
            })
            continue

        try:
            features = [float(v) for v in features]
            result = make_prediction(features)
            results.append({'index': i, **result})
        except Exception as e:
            errors.append({'index': i, 'error': str(e)})

    fraud_count = sum(1 for r in results if r.get('prediction') == 1)

    logger.info(
        f'/predict/batch | total={len(transactions)} '
        f'| success={len(results)} | fraud={fraud_count} | errors={len(errors)}'
    )

    response = {
        'results':     results,
        'total':       len(transactions),
        'processed':   len(results),
        'fraud_count': fraud_count,
        'error_count': len(errors)
    }
    if errors:
        response['errors'] = errors

    return jsonify(response), 200


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found', 'available': [
        'GET /health', 'GET /info', 'POST /predict', 'POST /predict/batch'
    ]}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed'}), 405


@app.errorhandler(500)
def server_error(e):
    logger.error(f'Unhandled error: {e}')
    return jsonify({'error': 'Internal server error'}), 500


# ── Entry point (local dev only — production uses gunicorn) ───────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
