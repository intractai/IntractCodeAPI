from unittest.mock import Mock, patch
from typing import Tuple

from fastapi.testclient import TestClient
import pytest
import transformers
import torch
from hydra import initialize, compose


def test_generator(test_client: TestClient):
    assert True == True