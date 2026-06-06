"""
tests/test_api.py — Smoke tests for the Fraud Detection API

Run locally:   python -m pytest tests/ -v
Run in CI:     pytest tests/test_api.py --tb=short

These tests use Flask's built-in test client — no server needs to be running.
Source: https://flask.palletsprojects.com/en/3.1.x/testing/
"""

import sys
import os
import json
import pytest

# Add parent directory to path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


@pytest.fixture
def client():
    """Create a Flask test client. Runs before each test."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# ── Health check ──────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get('/health')
    assert resp.status_code == 200


def test_health_returns_ok_status(client):
    resp = client.get('/health')
    data = resp.get_json()
    assert data['status'] == 'ok'


def test_health_includes_model_version(client):
    resp = client.get('/health')
    data = resp.get_json()
    assert 'model_version' in data


# ── Model info ────────────────────────────────────────────────────────────────

def test_info_returns_200(client):
    resp = client.get('/info')
    assert resp.status_code == 200


def test_info_has_required_fields(client):
    resp = client.get('/info')
    data = resp.get_json()
    required = ['model_version', 'optimal_threshold', 'feature_count',
                 'test_pr_auc', 'test_roc_auc']
    for field in required:
        assert field in data, f'Missing field: {field}'


# ── Single prediction ─────────────────────────────────────────────────────────

def _make_payload(n_features=30):
    """Helper: build a valid request payload with n_features zeros."""
    return {'features': [0.0] * n_features}


def test_predict_valid_request(client):
    resp = client.post(
        '/predict',
        data=json.dumps(_make_payload()),
        content_type='application/json'
    )
    assert resp.status_code == 200


def test_predict_response_has_required_fields(client):
    resp = client.post(
        '/predict',
        data=json.dumps(_make_payload()),
        content_type='application/json'
    )
    data = resp.get_json()
    required = ['fraud_probability', 'prediction', 'label',
                 'threshold_used', 'model_version']
    for field in required:
        assert field in data, f'Missing field: {field}'


def test_predict_probability_in_valid_range(client):
    resp = client.post(
        '/predict',
        data=json.dumps(_make_payload()),
        content_type='application/json'
    )
    prob = resp.get_json()['fraud_probability']
    assert 0.0 <= prob <= 1.0


def test_predict_label_is_valid(client):
    resp = client.post(
        '/predict',
        data=json.dumps(_make_payload()),
        content_type='application/json'
    )
    label = resp.get_json()['label']
    assert label in ('LEGITIMATE', 'FRAUD')


def test_predict_prediction_is_binary(client):
    resp = client.post(
        '/predict',
        data=json.dumps(_make_payload()),
        content_type='application/json'
    )
    pred = resp.get_json()['prediction']
    assert pred in (0, 1)


# ── Single prediction — error cases ──────────────────────────────────────────

def test_predict_missing_features_key(client):
    resp = client.post(
        '/predict',
        data=json.dumps({}),
        content_type='application/json'
    )
    assert resp.status_code == 400
    assert 'error' in resp.get_json()


def test_predict_wrong_feature_count(client):
    resp = client.post(
        '/predict',
        data=json.dumps({'features': [1.0, 2.0, 3.0]}),
        content_type='application/json'
    )
    assert resp.status_code == 400


def test_predict_non_numeric_features(client):
    resp = client.post(
        '/predict',
        data=json.dumps({'features': ['a', 'b'] + [0.0] * 28}),
        content_type='application/json'
    )
    assert resp.status_code == 400


def test_predict_invalid_json(client):
    resp = client.post(
        '/predict',
        data='not json at all',
        content_type='application/json'
    )
    assert resp.status_code == 400


def test_predict_features_not_list(client):
    resp = client.post(
        '/predict',
        data=json.dumps({'features': 'not a list'}),
        content_type='application/json'
    )
    assert resp.status_code == 400


# ── Batch prediction ──────────────────────────────────────────────────────────

def test_batch_valid_request(client):
    payload = {
        'transactions': [
            {'features': [0.0] * 30},
            {'features': [0.0] * 30},
            {'features': [0.0] * 30},
        ]
    }
    resp = client.post(
        '/predict/batch',
        data=json.dumps(payload),
        content_type='application/json'
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['total'] == 3
    assert data['processed'] == 3
    assert len(data['results']) == 3


def test_batch_response_fields(client):
    payload = {'transactions': [{'features': [0.0] * 30}]}
    resp = client.post(
        '/predict/batch',
        data=json.dumps(payload),
        content_type='application/json'
    )
    data = resp.get_json()
    assert 'results' in data
    assert 'total' in data
    assert 'fraud_count' in data


def test_batch_missing_transactions_key(client):
    resp = client.post(
        '/predict/batch',
        data=json.dumps({}),
        content_type='application/json'
    )
    assert resp.status_code == 400


def test_batch_exceeds_limit(client):
    payload = {'transactions': [{'features': [0.0] * 30}] * 1001}
    resp = client.post(
        '/predict/batch',
        data=json.dumps(payload),
        content_type='application/json'
    )
    assert resp.status_code == 400


# ── Wrong HTTP method ─────────────────────────────────────────────────────────

def test_get_predict_returns_405(client):
    resp = client.get('/predict')
    assert resp.status_code == 405


# ── 404 handler ───────────────────────────────────────────────────────────────

def test_unknown_route_returns_404(client):
    resp = client.get('/nonexistent')
    assert resp.status_code == 404
    assert 'error' in resp.get_json()
