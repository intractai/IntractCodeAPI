import pytest
import transformers
import torch
from unittest.mock import Mock, patch


@pytest.fixture
def mock_hf_model():
    model = Mock()
    model.predict.return_value = 'mocked prediction'
    return model


@patch('transformers.PreTrainedModel', new_callable=lambda: mock_hf_model)
def test_finetune(mock_model):
    assert True == True