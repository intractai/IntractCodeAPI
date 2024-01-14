from unittest.mock import Mock, patch
from typing import Tuple
import sys

from fastapi.testclient import TestClient
import pytest
import transformers
import torch
from hydra import initialize, compose

sys.path.append('/Users/erfanmiahi/projects/github/DockerAgent/')
from app.main import create_app
from src.modeling import ModelLoader, ModelProvider, get_model


class MockModelLoader(ModelLoader):
    def load_model(self) -> Tuple[torch.nn.Module, dict]:
        # Create a mock model
        mock_model = Mock(spec=torch.nn.Module)
        # You can set return values or side effects for specific methods if needed
        # For example: mock_model.predict.return_value = [0, 1, 1, 0]

        # Create a mock tokenizer or any other utility that should be in the returned dict
        mock_utils = {'tokenizer': Mock()}

        return mock_model, mock_utils


@pytest.fixture(scope="session")
def test_client():
    with initialize(version_base=None, config_path="../app/conf"):
        ModelProvider.register_model_loader('mock', MockModelLoader)
        cfg = compose(config_name="config", overrides=["model_cfg.model_type=mock"])
        app = create_app(cfg)
        with TestClient(app) as client:
            yield client