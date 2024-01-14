from typing import Tuple
from unittest.mock import Mock, patch

import pytest
import transformers
import torch
from fastapi.testclient import TestClient


def test_finetune(test_client: TestClient):
    assert True == True