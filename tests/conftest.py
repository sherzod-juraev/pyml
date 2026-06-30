import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_random_seed():
    """Test reproducibility for random seed."""
    np.random.seed(42)

def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'slow: marks tests as slow (deselect with -m "not slow")'
    )
