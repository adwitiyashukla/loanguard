"""Tests for evaluation metrics + cost-sensitive evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import binary_classification_metrics, ks_statistic, psi
from src.evaluation.business import (
    CostMatrix,
    cost_sensitive_evaluation,
    optimal_threshold,
)


@pytest.fixture
def perfect_labels():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    return y_true, y_proba


def test_metrics_perfect_separation(perfect_labels):
    y_true, y_proba = perfect_labels
    m = binary_classification_metrics(y_true, y_proba)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["ks"] == 1.0


def test_ks_zero_when_random():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=2000)
    y_proba = rng.random(size=2000)
    ks = ks_statistic(y_true, y_proba)
    assert ks < 0.1


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, a) < 1e-6


def test_psi_positive_for_drift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    b = rng.normal(loc=1.0, size=5000)  # shifted
    assert psi(a, b) > 0.1


def test_cost_sensitive_picks_minimum(perfect_labels):
    y_true, y_proba = perfect_labels
    t, c = optimal_threshold(y_true, y_proba, CostMatrix(false_negative=1000, false_positive=10))
    assert 0.0 < t < 1.0
    assert c >= 0
