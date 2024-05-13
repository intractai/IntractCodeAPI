
import logging
import pytest
import os, sys
# append the path to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from hydra import initialize, compose

from src import config_handler
from src.routers.fine_tuner import get_documentation_data


logger = logging.getLogger(__name__)
CONFIG_PATH = '../../src/conf'

# @pytest.mark.skip(reason="Skipping this test for now")
# @pytest.mark.parametrize("library", ['ninjax'])
def test_generate(library: str):

    # set of logging to log to the console

    # use phenix library
    # session = px.launch_app()

    initialize(config_path=CONFIG_PATH, job_name="test_app")
    config = compose(config_name="config")

    config_handler.ConfigProvider.initialize(config)
    print('documentation: ', get_documentation_data(library, True))


if __name__ == '__main__':
    test_generate('jax')