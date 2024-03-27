"""FastAPI server for running code completion model as a service."""

import logging
import sys

import hydra
import uvicorn
from omegaconf import DictConfig, OmegaConf
from fastapi import FastAPI

sys.path.append('../')
from src import modeling, config
from src.routers import generator, fine_tuner


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
def main(cfg: DictConfig):

    configure_logging()
    cfg = OmegaConf.create(cfg)
    modeling.ModelProvider(cfg.model)
    config.ConfigProvider.initialize(cfg)

    app.include_router(generator.router)
    app.include_router(fine_tuner.router)
    uvicorn.run(app, **cfg.server)


if __name__ == '__main__':
    main()
