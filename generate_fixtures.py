"""
generate_fixtures.py
---------------------
Run this ONCE after cloning to create minimal stub model artifacts
for the test suite. These are NOT real trained models — they are
deliberately simple objects that satisfy the same interface as the
production model so Flask's test client can load app.py without errors.

Usage:
    python generate_fixtures.py

Output:
    tests/fixtures/fraud_model.pkl
    tests/fixtures/scaler.pkl
    tests/fixtures/feature_cols.pkl
    tests/fixtures/metadata.json

Why this file exists:
    app.py loads model artifacts at import time using MODEL_DIR.
    conftest.py sets MODEL_DIR=tests/fixtures/ before app.py is imported.
    Without actual files in tests/fixtures/, every test fails with a
    FileNotFoundError before a single assertion runs.

    The production model/ directory is NOT committed to git (it contains
    large binary files). This script creates a lightweight alternative
    that is safe to commit.

Sources:
    sklearn DummyClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    joblib persistence:      https://joblib.readthedocs.io/en/stable/persistence.html
    pytest conftest:         https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-files
"""

import os
import json
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

# ── Output path ───────────────────────────────────────────────────────────────
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'tests', 'fixtures')
os.makedirs(FIXTURES_DIR, exist_ok=True)

# ── Feature columns: must match production exactly ────────────────────────────
# Order: V1–V28, then Time, then Amount (30 features total)
feature_cols = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']

# ── Stub model: DummyClassifier returns constant probabilities ────────────────
# strategy='constant' with constant=0 means predict_proba returns [[1.0, 0.0]]
# for every input — predictable, deterministic, interface-compatible.
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
X_dummy = np.zeros((10, 30))
y_dummy = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0])  # must have both classes

stub_model = DummyClassifier(strategy='prior', random_state=42)
stub_model.fit(X_dummy, y_dummy)

# Verify the interface matches what app.py expects
proba = stub_model.predict_proba(np.zeros((1, 30)))
assert proba.shape == (1, 2), f"predict_proba must return (1,2), got {proba.shape}"
print(f'Stub model predict_proba shape: {proba.shape} ✓')

# ── Stub scaler: fit on 2-column data (Time, Amount only) ────────────────────
# The production scaler is fit on X_train[['Time', 'Amount']] — 2 features.
# app.py extracts arr[:, [28, 29]] and passes it to scaler.transform().
# So the stub scaler must also be fit on 2-feature data.
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler = StandardScaler()
scaler.fit(np.array([[0.0, 0.0], [1.0, 100.0], [2.0, 50.0]]))  # 2 columns: Time, Amount
print(f'Stub scaler n_features_in_: {scaler.n_features_in_} ✓ (expected 2)')

# ── Metadata: matches metadata.json schema that app.py reads ─────────────────
metadata = {
    'model_version':       'test-fixture',
    'optimal_threshold':   0.5,
    'feature_count':       len(feature_cols),     # 30
    'cols_to_scale':       ['Time', 'Amount'],
    'training_fraud_rate': 0.1,
    'test_pr_auc':         0.85,
    'test_roc_auc':        0.97,
    'scale_pos_weight':    9.0,
    'description':         'Test fixture model — not for production'
}

# ── Write artifacts ───────────────────────────────────────────────────────────
joblib.dump(stub_model,    os.path.join(FIXTURES_DIR, 'fraud_model.pkl'))
joblib.dump(scaler,        os.path.join(FIXTURES_DIR, 'scaler.pkl'))
joblib.dump(feature_cols,  os.path.join(FIXTURES_DIR, 'feature_cols.pkl'))

with open(os.path.join(FIXTURES_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print()
print('─' * 50)
print('Test fixtures written to tests/fixtures/')
for fname in os.listdir(FIXTURES_DIR):
    size = os.path.getsize(os.path.join(FIXTURES_DIR, fname))
    print(f'  {fname:<25} {size/1024:.1f} KB')
print('─' * 50)
print('Run tests with: python -m pytest tests/ -v')
