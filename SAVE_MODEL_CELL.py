"""
─────────────────────────────────────────────────────────────────
ADD THIS AS THE LAST CELL IN YOUR NOTEBOOK (after Section 8 SHAP)
─────────────────────────────────────────────────────────────────
This serializes the trained model, scaler, threshold, and metadata
into the model/ folder so the Flask API can load them at startup.

Source: https://joblib.readthedocs.io/en/stable/persistence.html
"""

import joblib
import json
import os

# ── Create output folder ──────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)

# ── Serialize artifacts ───────────────────────────────────────────────────────
# joblib is recommended over pickle for numpy arrays and sklearn objects
# It is more efficient for large arrays (XGBoost trees stored as numpy internally)
joblib.dump(xgb_model,    'model/fraud_model.pkl')
joblib.dump(scaler,        'model/scaler.pkl')
joblib.dump(feature_cols,  'model/feature_cols.pkl')   # list of 30 feature names

# ── Save threshold and metadata as JSON (human-readable) ──────────────────────
metadata = {
    'model_version':    '1.0.0',
    'optimal_threshold': float(best_t),          # from threshold tuning cell
    'feature_count':     len(feature_cols),
    'cols_to_scale':     ['Time', 'Amount'],
    'training_fraud_rate': float(y_train.mean()),
    'test_pr_auc':       float(average_precision_score(y_test, y_prob_xgb)),
    'test_roc_auc':      float(roc_auc_score(y_test, y_prob_xgb)),
    'scale_pos_weight':  float(scale_pos_weight),
    'description':       'XGBoost fraud detector — MLG-ULB creditcard dataset'
}
with open('model/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('─' * 50)
print('Artifacts saved to model/')
for fname in os.listdir('model'):
    size = os.path.getsize(f'model/{fname}')
    print(f'  {fname:<25} {size/1024:.1f} KB')
print('─' * 50)
print(f'Optimal threshold: {best_t:.2f}')
print(f'Test PR-AUC:       {metadata["test_pr_auc"]:.4f}')
print('Model ready for deployment.')
