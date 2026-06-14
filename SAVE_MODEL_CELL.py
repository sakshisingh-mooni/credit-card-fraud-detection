"""
─────────────────────────────────────────────────────────────────
ADD THIS AS THE LAST CELL IN YOUR NOTEBOOK (after Section 8 SHAP)
─────────────────────────────────────────────────────────────────
Builds a single sklearn Pipeline (ColumnTransformer + XGBClassifier),
serialises it as one .joblib artifact, and writes metadata.json.

Why a Pipeline instead of separate scaler + model files?
  - Preprocessing and inference are always applied as a unit.
  - No train/serve skew: the scaler's fitted statistics live inside
    the pipeline and travel with the model.
  - One file to version, deploy, and load — nothing to forget.

Source: https://scikit-learn.org/stable/modules/pipeline.html
Source: https://scikit-learn.org/stable/modules/compose.html#columntransformer
Source: https://joblib.readthedocs.io/en/stable/persistence.html
"""

import joblib, json, os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

os.makedirs('model', exist_ok=True)

# ── Re-split to get raw (unscaled) features ───────────────────────────────────
# Section 4 scaled X_train in-place on the DataFrame. The Pipeline must receive
# raw features so its ColumnTransformer can learn Time/Amount statistics itself.
# Same random_state + stratify guarantees identical train/test rows as Section 4.
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f'Raw split: {len(X_train_raw):,} train | {len(X_test_raw):,} test')

# ── Build Pipeline: ColumnTransformer + XGBClassifier ────────────────────────
# ColumnTransformer applies StandardScaler to Time and Amount only (indices 28, 29).
# remainder='passthrough' leaves V1-V28 untouched (already PCA-scaled at source).
cols_to_scale = ['Time', 'Amount']
scale_indices = [feature_cols.index(c) for c in cols_to_scale]  # [28, 29]

preprocessor = ColumnTransformer(
    transformers=[
        ('scale_time_amount', StandardScaler(), scale_indices),
    ],
    remainder='passthrough',
    verbose_feature_names_out=False,
)

fraud_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   xgb_model),
])

# ── Fit on raw training data ──────────────────────────────────────────────────
fraud_pipeline.fit(X_train_raw, y_train_raw)
print('Pipeline fitted.')

# ── Verify pipeline output matches standalone model ───────────────────────────
pipeline_probs    = fraud_pipeline.predict_proba(X_test_raw)[:, 1]
pr_auc_pipeline   = average_precision_score(y_test_raw, pipeline_probs)
pr_auc_standalone = average_precision_score(y_test, y_prob_xgb)
print(f'Pipeline PR-AUC   : {pr_auc_pipeline:.4f}')
print(f'Standalone PR-AUC : {pr_auc_standalone:.4f}')
assert abs(pr_auc_pipeline - pr_auc_standalone) < 0.01, \
    f"Pipeline PR-AUC {pr_auc_pipeline:.4f} diverges from standalone {pr_auc_standalone:.4f}"
print('Pipeline output verified ✓')

# ── Serialise single artifact ─────────────────────────────────────────────────
joblib.dump(fraud_pipeline, 'model/fraud_pipeline.joblib')

metadata = {
    'model_version':      '2.0.0',
    'artifact':           'fraud_pipeline.joblib',
    'pipeline_steps':     ['preprocessor (ColumnTransformer)', 'classifier (XGBClassifier)'],
    'optimal_threshold':   float(best_t),
    'feature_count':       len(feature_cols),
    'feature_order':       feature_cols,
    'cols_to_scale':       cols_to_scale,
    'training_fraud_rate': float(y_train_raw.mean()),
    'test_pr_auc':         float(pr_auc_pipeline),
    'test_roc_auc':        float(roc_auc_score(y_test_raw, pipeline_probs)),
    'scale_pos_weight':    float(scale_pos_weight),
    'description':         'XGBoost fraud detector — MLG-ULB creditcard dataset — sklearn Pipeline v2'
}
with open('model/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('─' * 50)
print('Artifacts saved to model/')
for fname in sorted(os.listdir('model')):
    size = os.path.getsize(f'model/{fname}')
    print(f'  {fname:<30} {size/1024:.1f} KB')
print('─' * 50)
print(f'Optimal threshold : {best_t:.2f}')
print(f'Pipeline PR-AUC   : {metadata["test_pr_auc"]:.4f}')
print(f'Pipeline ROC-AUC  : {metadata["test_roc_auc"]:.4f}')
print('Single pipeline artifact ready for deployment.')
