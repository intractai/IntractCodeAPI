"""FastAPI server for running code completion model as a service."""

from typing import Dict
import logging
import os
import sys
from contextlib import asynccontextmanager

import hydra
import uvicorn
from omegaconf import DictConfig, OmegaConf
from fastapi import FastAPI

sys.path.append('../')
from src import modeling, config
from src.arg_handling import parse_args
from app.routers import generator, fine_tuner


logger = logging.getLogger(__name__)


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
    yield
    logger.info('finished')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the model when the server starts up."""
    logger.info('Lifespan Started!')
    cfg = config.get_config()
    logger.info(cfg)
    modeling.ModelProvider(cfg.model_cfg)
    yield
    logger.info('LifeSpan Finished!')


def pre_app_init(cfg: DictConfig):
    configure_logging()
    cfg = OmegaConf.create(cfg)
    config.ConfigProvider.initialize(cfg)


def create_app(cfg: DictConfig):
    pre_app_init(cfg)
    app = FastAPI(lifespan=lifespan)
    app.include_router(generator.router)
    app.include_router(fine_tuner.router)
    return app


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    app = create_app(cfg)
    uvicorn.run(app, **cfg.server_cfg)
    
if __name__ == '__main__':
    main()
