"""FastAPI server for running code completion model as a service."""

import logging
import sys

import hydra
import uvicorn
from omegaconf import DictConfig, OmegaConf
from fastapi import FastAPI

sys.path.append('../')
from src import modeling, config_handler, database
from src.routers import auth, fine_tuner, generator


logger = logging.getLogger(__name__)
app = FastAPI()


def configure_logging():
    """Configure logging for the server."""
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # add log in the file
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_app() -> FastAPI:
    return app


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):

    configure_logging()
    config = OmegaConf.create(config)
    modeling.ModelProvider(config.model)
    config_handler.ConfigProvider.initialize(config)
    database.DatabaseProvider.initialize(config.database.path)


    app.include_router(auth.router)
    app.include_router(fine_tuner.router)
    app.include_router(generator.router)
    uvicorn.run(app, **config.server)


if __name__ == '__main__':
    main()
