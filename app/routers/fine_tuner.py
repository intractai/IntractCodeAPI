from typing import Dict, Annotated
from argparse import Namespace
import logging
import threading

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, CancelledError

from src import modeling, finetune
from src import config

logger = logging.getLogger(__name__)
router = APIRouter()


class ProjectFinetuneData(BaseModel):
    project_dict: Dict[str, str]


@router.post("/finetune/project")
def finetune_project(item: ProjectFinetuneData, cfg: Annotated[Namespace, Depends(config.get_config)]):
    # for file_name, file_code in item.project_dict.items():
    #     logger.info(f">>> {file_name}\n\n{file_code}\n\n")

    # item.project_dict = {k: v for k, v in item.project_dict.items()
    #                 if not k.startswith('ninjax') \
    #                     or k in ['ninjax/examples/quickstart.py', 'ninjax/examples/libraries.py', 'ninjax/ninjax/ninjax.py']}

    try: 
        with ThreadPoolExecutor() as executor:
            # Create a new future for the incoming request
            job_thread = executor.submit(finetune_task, item, cfg)
            # Run the future and return the result
            result = job_thread.result()
    except CancelledError:
        logger.info("Cancelled /generate execution by a new request.")

    # finetune_args = env_args['finetune']
    # finetune.train_supervised_projectdir(
    #     item.project_dict, output_dir=local_model_dir,
    #     report_to='none', **vars(finetune_args))

    return {"result": "success"}


def finetune_task(item: ProjectFinetuneData, cfg: Namespace):
    # Print the current thread id to show that it is different for each request
    modeling.GLOBAL_FINETUNE_THREAD_ID = threading.get_ident()

    model_cfg = cfg.model_cfg
    train_cfg = cfg.train_cfg
    finetune.train_supervised_projectdir(
        item.project_dict, output_dir=model_cfg.save_model_dir,
        report_to='none', train_cfg=train_cfg)

    return {"result": "success"}
