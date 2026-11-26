from pathlib import Path
import pytest
from bayesian_nemo import BayesianPolicy

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='session')
def policy():
    return BayesianPolicy.from_yaml(ROOT/'examples/policy.yaml')

@pytest.fixture(scope='session')
def corrected_policy():
    return BayesianPolicy.from_yaml(ROOT/'examples/corrected_policy.yaml')
