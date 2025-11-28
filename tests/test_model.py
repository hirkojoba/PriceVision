"""
Unit tests for model module
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import (
    time_series_train_test_split,
    train_model,
    evaluate_model,
    get_feature_importance,
    save_model,
    load_model
)


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    np.random.seed(42)
    n_samples = 200

    # Create features
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
        'feature_4': np.random.randn(n_samples),
        'feature_5': np.random.randn(n_samples),
    })

    # Create labels (random but deterministic)
    y = pd.Series(np.random.choice([-1, 0, 1], size=n_samples, p=[0.3, 0.4, 0.3]))

    return X, y


def test_time_series_train_test_split(sample_dataset):
    """Test time series split."""
    X, y = sample_dataset

    X_train, X_test, y_train, y_test = time_series_train_test_split(
        X, y, test_size=0.2
    )

    # Check sizes
    assert len(X_train) == int(len(X) * 0.8)
    assert len(X_test) == len(X) - len(X_train)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Check chronological order (first part is train)
    assert X_train.index[-1] < X_test.index[0]


def test_train_model_random_forest(sample_dataset):
    """Test Random Forest training."""
    X, y = sample_dataset
    X_train, X_test, y_train, y_test = time_series_train_test_split(X, y)

    model = train_model(X_train, y_train, model_type="random_forest")

    # Check model was trained
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    assert hasattr(model, 'feature_importances_')

    # Check predictions work
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
    assert all(pred in [-1, 0, 1] for pred in predictions)


def test_train_model_gradient_boosting(sample_dataset):
    """Test Gradient Boosting training."""
    X, y = sample_dataset
    X_train, X_test, y_train, y_test = time_series_train_test_split(X, y)

    model = train_model(
        X_train, y_train,
        model_type="gradient_boosting",
        n_estimators=50
    )

    # Check model was trained
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')

    # Check predictions work
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)


def test_evaluate_model(sample_dataset):
    """Test model evaluation."""
    X, y = sample_dataset
    X_train, X_test, y_train, y_test = time_series_train_test_split(X, y)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    # Check metrics keys
    assert 'accuracy' in metrics
    assert 'confusion_matrix' in metrics
    assert 'classification_report' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics

    # Check accuracy is in valid range
    assert 0 <= metrics['accuracy'] <= 1

    # Check confusion matrix shape
    cm = metrics['confusion_matrix']
    assert cm.shape[0] == cm.shape[1]  # Square matrix

    # Check per-class metrics
    for class_name in ['DOWN', 'FLAT', 'UP']:
        assert class_name in metrics['precision']
        assert class_name in metrics['recall']
        assert class_name in metrics['f1_score']


def test_get_feature_importance(sample_dataset):
    """Test feature importance extraction."""
    X, y = sample_dataset
    X_train, X_test, y_train, y_test = time_series_train_test_split(X, y)

    model = train_model(X_train, y_train)
    importance_df = get_feature_importance(model, X.columns.tolist())

    # Check DataFrame structure
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns

    # Check number of features
    assert len(importance_df) == len(X.columns)

    # Check importance values are valid
    assert all(importance_df['importance'] >= 0)
    assert np.isclose(importance_df['importance'].sum(), 1.0)

    # Check sorted in descending order
    assert all(importance_df['importance'].iloc[i] >= importance_df['importance'].iloc[i+1]
               for i in range(len(importance_df) - 1))


def test_save_and_load_model(sample_dataset):
    """Test model persistence."""
    X, y = sample_dataset
    X_train, X_test, y_train, y_test = time_series_train_test_split(X, y)

    # Train model
    model = train_model(X_train, y_train)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        model_path = tmp_file.name

    try:
        # Save model
        feature_names = X.columns.tolist()
        save_model(model, model_path, feature_names)

        # Check file exists
        assert os.path.exists(model_path)

        # Load model
        loaded_model, loaded_features = load_model(model_path)

        # Check loaded model works
        predictions_original = model.predict(X_test)
        predictions_loaded = loaded_model.predict(X_test)

        # Check predictions match
        assert all(predictions_original == predictions_loaded)

        # Check feature names match
        assert loaded_features == feature_names

    finally:
        # Cleanup
        if os.path.exists(model_path):
            os.unlink(model_path)


def test_model_with_custom_parameters(sample_dataset):
    """Test training with custom parameters."""
    X, y = sample_dataset
    X_train, X_test, y_train, y_test = time_series_train_test_split(X, y)

    model = train_model(
        X_train, y_train,
        model_type="random_forest",
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=123
    )

    # Check model parameters were set
    assert model.n_estimators == 100
    assert model.max_depth == 10
    assert model.min_samples_split == 10
    assert model.random_state == 123


def test_invalid_model_type(sample_dataset):
    """Test that invalid model type raises error."""
    X, y = sample_dataset
    X_train, _, y_train, _ = time_series_train_test_split(X, y)

    with pytest.raises(ValueError):
        train_model(X_train, y_train, model_type="invalid_model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
