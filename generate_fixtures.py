"""
generate_fixtures.py
---------------------
Run this ONCE after cloning to create a minimal stub pipeline artifact
for the test suite. The stub is NOT a real trained model — it is a
deliberately simple sklearn Pipeline that satisfies the same interface
as the production artifact so Flask's test client can load app.py
without errors.

Usage:
    python generate_fixtures.py

Output:
    tests/fixtures/fraud_pipeline.joblib
    tests/fixtures/metadata.json

Why this file exists:
    app.py loads model artifacts at import time using MODEL_DIR.
    conftest.py sets MODEL_DIR=tests/fixtures/ before app.py is imported.
    Without actual files in tests/fixtures/, every test fails with a
    FileNotFoundError before a single assertion runs.

    The production model/ directory is NOT committed to git (large binary).
    This script creates a lightweight alternative safe to commit.

Sources:
    sklearn Pipeline:      https://scikit-learn.org/stable/modules/pipeline.html
    sklearn ColumnTransformer: https://scikit-learn.org/stable/modules/compose.html
    DummyClassifier:       https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    joblib persistence:    https://joblib.readthedocs.io/en/stable/persistence.html
    pytest conftest:       https://docs.pytest.org/en/stable/reference/fixtures.html
"""

import os
import json
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

# ── Output path ───────────────────────────────────────────────────────────────
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'tests', 'fixtures')
os.makedirs(FIXTURES_DIR, exist_ok=True)

# ── Feature columns: must match production exactly ────────────────────────────
# Order: V1–V28, then Time, then Amount (30 features total)
feature_cols = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']

# Time=index 28, Amount=index 29
cols_to_scale = ['Time', 'Amount']
scale_indices = [feature_cols.index(c) for c in cols_to_scale]  # [28, 29]

# ── Stub Pipeline: ColumnTransformer + DummyClassifier ───────────────────────
# Mirrors the production Pipeline structure exactly so app.py loads without errors.
# DummyClassifier(strategy='prior') returns constant class probabilities —
# predictable and deterministic for tests.
preprocessor = ColumnTransformer(
    transformers=[
        ('scale_time_amount', StandardScaler(), scale_indices),
    ],
    remainder='passthrough',
    verbose_feature_names_out=False,
)

stub_classifier = DummyClassifier(strategy='prior', random_state=42)

stub_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   stub_classifier),
])

# Fit on minimal dummy data — must have both classes for DummyClassifier
X_dummy = np.zeros((10, 30))
y_dummy = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
stub_pipeline.fit(X_dummy, y_dummy)

# Verify the interface matches what app.py expects
proba = stub_pipeline.predict_proba(np.zeros((1, 30)))
assert proba.shape == (1, 2), f"predict_proba must return (1, 2), got {proba.shape}"
print(f'Stub pipeline predict_proba shape: {proba.shape} ✓')
print(f'Pipeline steps: {[name for name, _ in stub_pipeline.steps]} ✓')

# ── Metadata: matches metadata.json schema that app.py reads ─────────────────
metadata = {
    'model_version':      'test-fixture',
    'artifact':           'fraud_pipeline.joblib',
    'pipeline_steps':     ['preprocessor (ColumnTransformer)', 'classifier (DummyClassifier)'],
    'optimal_threshold':   0.5,
    'feature_count':       len(feature_cols),   # 30
    'feature_order':       feature_cols,
    'cols_to_scale':       cols_to_scale,
    'training_fraud_rate': 0.1,
    'test_pr_auc':         0.85,
    'test_roc_auc':        0.97,
    'scale_pos_weight':    9.0,
    'description':         'Test fixture pipeline — not for production'
}

# ── Write artifacts ───────────────────────────────────────────────────────────
joblib.dump(stub_pipeline, os.path.join(FIXTURES_DIR, 'fraud_pipeline.joblib'))

with open(os.path.join(FIXTURES_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print()
print('─' * 50)
print('Test fixtures written to tests/fixtures/')
for fname in os.listdir(FIXTURES_DIR):
    size = os.path.getsize(os.path.join(FIXTURES_DIR, fname))
    print(f'  {fname:<30} {size/1024:.1f} KB')
print('─' * 50)
print('Run tests with: python -m pytest tests/ -v')
