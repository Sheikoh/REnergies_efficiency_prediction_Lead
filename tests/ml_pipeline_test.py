import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from app.train import load_data, preprocess_data, create_pipeline, train_model

# ==========================================
# 1. FIXTURES (Reusable Dummy Data)
# ==========================================

@pytest.fixture
def mock_df():
    """
    Creates a small, deterministic DataFrame that mimics the real data structure.
    This allows us to test logic without downloading 2MB of CSV files.
    """
    # Create 10 rows of dummy data with the same dimensions as typical housing data
    data = {
        'MedInc': np.random.rand(10),
        'HouseAge': np.random.randint(10, 50, 10),
        'AveRooms': np.random.rand(10) * 5,
        'AveBedrms': np.random.rand(10) * 2,
        'Population': np.random.randint(100, 1000, 10),
        'AveOccup': np.random.rand(10) * 3,
        'Latitude': np.random.rand(10) * 50,
        'Longitude': np.random.rand(10) * 50,
        'MedHouseVal': np.random.rand(10) * 100000 # The Target
    }
    return pd.DataFrame(data)

# ==========================================
# 2. UNIT TESTS
# ==========================================

def test_load_data(mock_df):
    """
    Test loading data. We patch 'pd.read_csv' so it returns our 
    mock_df instead of trying to go to the internet.
    """
    with patch('app.train.pd.read_csv') as mock_read_csv:
        mock_read_csv.return_value = mock_df
        
        # We can pass any string here, it won't actually hit the URL
        df = load_data("http://fake-url.com/data.csv")
        
        assert not df.empty
        assert df.shape == (10, 9) # 9 columns (8 features + 1 target)
        mock_read_csv.assert_called_once()

def test_preprocess_data(mock_df):
    """
    Test that data splits correctly into 80/20 train/test.
    """
    X_train, X_test, y_train, y_test = preprocess_data(mock_df, test_size=0.2)
    
    # Check dimensions
    assert len(X_train) == 8  # 80% of 10 rows
    assert len(X_test) == 2   # 20% of 10 rows
    
    # Check that Target column is removed from Features (X)
    assert 'MedHouseVal' not in X_train.columns
    assert 'MedHouseVal' not in X_test.columns

def test_create_pipeline():
    """
    Test that the pipeline contains the expected steps.
    """
    pipe = create_pipeline()
    
    # Check object type
    assert isinstance(pipe, Pipeline)
    
    # Check steps existence
    assert "standard_scaler" in pipe.named_steps
    assert "Random_Forest" in pipe.named_steps
    
    # Check step types (Robustness check)
    assert isinstance(pipe.named_steps['Random_Forest'], RandomForestRegressor)

def test_train_model_smoke_test(mock_df):
    """
    A 'Smoke Test'. Instead of mocking fit(), we actually run it on 
    the tiny mock_df. This catches bugs in parameter names that mocks miss.
    """
    # 1. Prepare data
    X_train, X_test, y_train, y_test = preprocess_data(mock_df)
    pipe = create_pipeline()
    
    # 2. Configure a minimal grid to make the test run instantly
    # We use 1 estimator and 2 CV folds (since we only have 8 training rows)
    param_grid = {
        "Random_Forest__n_estimators": [1], 
        "Random_Forest__criterion": ["squared_error"]
    }
    
    # 3. Run Training
    model = train_model(pipe, X_train, y_train, param_grid, cv=2)
    
    # 4. Assertions
    assert model.best_estimator_ is not None
    # Ensure it actually learned something (even if nonsense on random data)
    assert hasattr(model.best_estimator_, 'predict')