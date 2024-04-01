
from src.routers.fine_tuner import get_documentation_data
import logging
import sys
import pytest

from hydra.experimental import initialize, compose
import uvicorn
from omegaconf import DictConfig, OmegaConf
from fastapi import FastAPI
from src import config_handler, database

CONFIG_PATH = '../../src/conf'

@pytest.mark.skip(reason="Skipping this test for now")
@pytest.mark.parametrize("library", ['ninjax'])
def test_generate(library: str):
    initialize(config_path=CONFIG_PATH, config_name='config', job_name="test_app")
    config = compose(config_name="config")

    config_handler.ConfigProvider.initialize(config)
    print(get_documentation_data(library, True))

