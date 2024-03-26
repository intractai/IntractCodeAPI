from typing import Annotated, Dict, List
from argparse import Namespace
import logging
import threading

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, CancelledError

from src import config, modeling
from src.training import finetune

logger = logging.getLogger(__name__)
router = APIRouter()


class ProjectFinetuneData(BaseModel):
    project_dict: Dict[str, str]
    language: str = 'python'
    libraries: List[str] = []


@router.post("/finetune/project")
def finetune_project(item: ProjectFinetuneData, cfg: Annotated[Namespace, Depends(config.get_config)]):
    try: 
        with ThreadPoolExecutor() as executor:
            # Create a new future for the incoming request
            job_thread = executor.submit(finetune_task, item, cfg)
            # Run the future and return the result
            result = job_thread.result()
    except CancelledError:
        logger.info("Cancelled /generate execution by a new request.")

    # finetune_args = env_args['finetune']
    # finetune.train_self_supervised_project(
    #     item.project_dict, output_dir=local_model_dir,
    #     report_to='none', **vars(finetune_args))

    return {"result": "success"}


def get_documentation_data(
        library: str, gen_problems: bool = False
    ) -> Dict[str, List[str]]:
    """Get the documentation data for a library.
    
    Args:
        library (str): The library to get documentation data for.
        gen_problems (bool): Whether to generate problems. Defaults to False.
        
    Returns:
        Dict[str, List[str]]: The documentation data.
    """
    return_data = {
        'text': [],
        'code': [],
    }

    if gen_problems:
        return_data['problems'] = []

    return return_data




# Finetune steps:
#   1. Collect documentation for each library
#   2. Finetune the model with NTP and FIM on the collected documentation
#   3. Finetune the model with NTP and FIM on code snippets for each library (collected from documentation)
#   4. Generate practice problems and ideal solutions for each library
#   5. Instruction finetune on the generated practice problems
#   6. Finetune the model with NTP and FIM on the verified solutions
#
# Options:
#   - train_on_code
#   - train_on_docs
#   - train_on_doc_code
#   - train_on_practice_problems (uses instruction finetuning)
#   - train_on_verified_solutions
#
#   - use_ntp (next token prediction)
#   - use_fim (fill-in-the-middle)

def finetune_task(item: ProjectFinetuneData, cfg: Namespace):
    # Print the current thread id to show that it is different for each request
    modeling.GLOBAL_FINETUNE_THREAD_ID = threading.get_ident()
    model = cfg.model
    trainer_cfg = cfg.train.trainer

    # Determine what data to collect / generate
    require_problems = cfg.train.train_on_practice_problems \
        or cfg.train.train_on_verified_solutions
    require_docs = cfg.train.train_on_docs \
        or cfg.train.train_on_doc_code \
        or require_problems
    
    train_methods = []
    if cfg.train.use_ntp:
        train_methods.append('ntp')
    if cfg.train.use_fim:
        train_methods.append('fim')
    
    # Get the documentation data for each library
    if require_docs:
        n_workers = cfg.train.max_retrieval_threads
        executor = ThreadPoolExecutor(max_workers=n_workers)
        lib_futures = {}
        for library in item.libraries:
            lib_futures[library] = executor.submit(
                get_documentation_data, library, require_problems)

    # Train on the user's codebase while asynchronously collecting documentation
    if cfg.train.train_on_code:
        logger.info("Training on user's codebase")
        finetune.train_self_supervised_project(
            item.project_dict, output_dir=model.save_model_dir,
            report_to='none', train_cfg=trainer_cfg, train_methods=train_methods, # TODO: Test train_methods
        )
        
    # Train on the collected documentation
    if require_docs:
        lib_data = {}

        # Wait for all the documentation data to be collected
        for library, lib_future in lib_futures.items():
            lib_data[library] = lib_future.result()

    # Train on the documentation text
    if cfg.train.train_on_docs:
        train_documents = []
        for lib_data in lib_data.values():
            train_documents.extend(lib_data['text'])

        logger.info("Training on documentation text")
        finetune.train_self_supervised( # TODO: Test train_self_supervised
            model, train_documents, output_dir=model.save_model_dir,
            report_to='none', train_cfg=cfg.train.trainer, train_methods=['ntp'])
            
    # Train on the documentation code snippets
    if cfg.train.train_on_doc_code:
        train_code = []
        for lib_data in lib_data.values():
            train_documents.extend(lib_data['code'])

        logger.info("Training on documentation code snippets")
        finetune.train_self_supervised(
            model, train_code, output_dir=model.save_model_dir,
            report_to='none', train_cfg=cfg.train.trainer, train_methods=train_methods)
            
    # Generate practice problems and ideal solutions
    if require_problems:
        train_problems = []
        for library, lib_data in lib_data.items():
            train_problems.extend(lib_data['problems'])

        if cfg.train.train_on_practice_problems:
            logger.info("Training on practice problems")
            solutions = interactive.multi_step_sft.train_sft_with_verification( # TODO: Add train_sft_with_verification
                model, train_problems, output_dir=model.save_model_dir,
                report_to='none', train_cfg=cfg.train.trainer)
        else:
            logger.info("Generating solutions to practice problems")
            solutions = interactive.multi_step.generate_solutions( # TODO: Add generate_solutions
                model, train_problems, output_dir=model.save_model_dir,
                report_to='none', train_cfg=cfg.train.trainer)
        
        # Train on the verified solutions
        if cfg.train.train_on_verified_solutions:
            finetune.train_self_supervised(
                model, solutions, output_dir=model.save_model_dir,
                report_to='none', train_cfg=cfg.train.trainer, train_methods=train_methods)


    return {"result": "success"}
