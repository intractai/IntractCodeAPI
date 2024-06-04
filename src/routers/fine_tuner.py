from functools import partial
import logging
import threading
from typing import Annotated, Dict, List, Optional

from concurrent.futures import ThreadPoolExecutor, CancelledError
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from omegaconf import DictConfig
from pydantic import BaseModel
import torch
from transformers import PreTrainedTokenizer

from src import config_handler
from src.auto_generation.problem_generator import LibraryProblemGenerator
from src.crawler.docs_scraper import get_doc_data
from src.modeling import get_model, get_tokenizer
from src.training import finetune
from src.training.interactive.train_multi_step_sft import (
    generate_solutions,
    train_multi_step_sft_with_verification,
)
from src.users import SessionTracker, validate_user_session


FINETUNE_ACTIVITY = 'finetune'
FINETUNE_THREAD_IDS = set()

logger = logging.getLogger(__name__)
router = APIRouter()
global_finetune_lock = threading.Lock()


class ProjectFinetuneData(BaseModel):
    project_dict: Optional[Dict[str, str]] = None
    language: Optional[str] = None
    libraries: Optional[List[str]] = None
    urls: Optional[List[str]] = None


# @router.get('/learn/project')
# async def finetune_project_form(
#         username: Annotated[str, Depends(validate_user_session)],
#     ):
#     # Send the static/train_page/train.html file with the following data:
#     # username, project path, whether or not the user is already finetuning a project
#     sess_tracker = SessionTracker.get_instance()
#     finetune_in_progress = sess_tracker.is_user_activity_running(username, FINETUNE_ACTIVITY)
#     return FileResponse(
#         'static/train_page/train.html',
#         media_type = 'text/html',
#         headers = {
#             'username': username,
#             'finetune_in_progress': str(finetune_in_progress)
#         }
#     )


@router.post('/finetune/project')
def finetune_project(
        item: ProjectFinetuneData,
        config: Annotated[DictConfig, Depends(config_handler.get_config)],
        username: Annotated[str, Depends(partial(validate_user_session, activity=FINETUNE_ACTIVITY))],
    ):
    max_threads = config.session.max_finetune_threads
    thread_id = threading.get_ident()

    try:
        verify_request_data(item)
    except ValueError as e:
        return {'result': 'error', 'message': str(e)}

    try:
        started = False
        if len(FINETUNE_THREAD_IDS) < max_threads:
            with global_finetune_lock:
                if len(FINETUNE_THREAD_IDS) < max_threads:
                    started = True
                    FINETUNE_THREAD_IDS.add(thread_id)

        if started:
            result = finetune_task(item, config, username)
        else:
            result = {
                'result': 'error',
                'message': 'Server training capacity reached. Try again later.'
            }
    except CancelledError:
        logger.info("Cancelled /fintune/project execution for a new request.")
        result = {'result': 'cancelled'}
    finally:
        if thread_id in FINETUNE_THREAD_IDS:
            FINETUNE_THREAD_IDS.remove(thread_id)

    return result


def verify_request_data(item: ProjectFinetuneData):
    if not (item.project_dict or item.libraries or item.urls):
        raise ValueError("Either project_dict or libraries must be provided.")
    elif item.libraries and not item.language:
        raise ValueError("Language must be provided if libraries are provided.")


def get_documentation_data(
        library: Optional[str] = None, url: Optional[str] = None,
        gen_problems: bool = False, lang: str = 'Python', feature_num: int = 10,
        problem_num_per_bullet_point: int = 30, max_char_count: int = 10000,
        model: str = 'gpt-3.5-turbo',
    ) -> Dict[str, List[str]]:
    """Get the documentation data for a library.
    
    Args:
        library (str): The library to get documentation data for.
        gen_problems (bool): Whether to generate problems. Defaults to False.
        
    Returns:
        Dict[str, List[str]]: The documentation data.
    """
    return_data = get_doc_data(library, url, lang)
    
    # Remove duplicate strings
    return_data['content'] = list(set(return_data['content']))
    return_data['code'] = list(set(return_data['code']))

    if gen_problems:
        lib_problem_generator = LibraryProblemGenerator(
            model, lang, library, url, max_char_count,
            feature_num, problem_num_per_bullet_point,
        )
        return_data['problems'] = lib_problem_generator.generate()

    return return_data


def finetune_task(item: ProjectFinetuneData, config: DictConfig, username: str) :
    """Finetune the model with the user's codebase and documentation.
    
    Args:
        item (ProjectFinetuneData): The project, language, and library data.
        config (DictConfig): The global config.
        username (str): The username of the user.
    """
    logging.info(f"Finetuning on project for user: {username}.")
    item.libraries = item.libraries or []

    model = get_model(username)
    tokenizer = get_tokenizer(username)

    finetune_model(item, config, model, tokenizer)

    return {"result": "success"}


def finetune_model(
        item: ProjectFinetuneData,
        config: DictConfig,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer
    ) -> torch.nn.Module:
    """Finetune the model with the user's codebase and documentation.
    
    Finetune steps:
        1. Collect documentation for each library
        2. Finetune the model with NTP and FIM on the collected documentation
        3. Finetune the model with NTP and FIM on code snippets for each library (collected from documentation)
        4. Generate practice problems and ideal solutions for each library
        5. Instruction finetune on the generated practice problems
        6. Finetune the model with NTP and FIM on the verified solutions

    Options (in config):
        - train_on_code
        - train_on_docs
        - train_on_doc_code
        - train_on_practice_problems (uses instruction finetuning)
        - train_on_verified_solutions

        - use_ntp (next token prediction)
        - use_fim (fill-in-the-middle)
    
    Args:
        item (ProjectFinetuneData): The project, language, libraries, and urls.
        config (DictConfig): The global config.
        model (torch.nn.Module): The model to finetune.
        tokenizer (PreTrainedTokenizer): The tokenizer for the model.
    """
    model_config = config.model
    
    # Determine what data to collect / generate
    require_problems = config.train.train_on_practice_problems \
        or config.train.train_on_verified_solutions
    require_docs = (
        config.train.train_on_docs \
        or config.train.train_on_doc_code \
        or require_problems
    ) and (
        bool(item.libraries)
        or bool(item.urls)
    )
        
    
    train_methods = []
    if config.train.use_ntp:
        train_methods.append('ntp')
    if config.train.use_fim:
        train_methods.append('fim')
    
    # Get the documentation data for each library
    if require_docs:
        n_workers = config.train.max_retrieval_threads
        executor = ThreadPoolExecutor(max_workers=n_workers)
        lib_futures = {}
        for library in item.libraries:
            lib_futures[library] = executor.submit(
                get_documentation_data, library=library, gen_problems=require_problems)
        for url in item.urls:
            lib_futures[url] = executor.submit(
                get_documentation_data, url=url, gen_problems=require_problems)

    # Train on the user's codebase while asynchronously collecting documentation
    if config.train.train_on_code and item.project_dict:
        logger.info("Training on user's codebase")
        finetune.train_self_supervised_project(
            model, tokenizer, item.project_dict, config.train,
            output_dir=model_config.save_model_dir, report_to='none',
            train_methods=train_methods,
        )

    # Train on the collected documentation
    if require_docs:
        lib_data = {}

        # Wait for all the documentation data to be collected
        for library, lib_future in lib_futures.items():
            result = lib_future.result()
            if result:
                lib_data[library] = result
            else:
                logger.error(f"Failed to get documentation data for '{library}'!")

        # If no documentation data was collected, return an error
        if not lib_data:
            return {"result": "error", "message": "Failed to get documentation data."}

        # Train on the documentation text
        if config.train.train_on_docs:
            train_documents = []
            for data in lib_data.values():
                train_documents.extend(data['content'])

            if len(train_documents) == 0:
                logger.info("No documentation content found for training, skipping.")
            else:
                logger.info(f"Training on {len(train_documents)} documents of documentation text")
                finetune.train_self_supervised_documents(
                    model, tokenizer, train_documents, config.train,
                    output_dir=model_config.save_model_dir, report_to='none',
                    train_methods=['ntp'],
                )
                
        # Train on the documentation code snippets

        if config.train.train_on_doc_code:
            train_code = []
            for data in lib_data.values():
                train_code.extend(data['code'])

            if len(train_code) == 0:
                logger.info("No documentation code snippets found for training, skipping.")
            else:
                logger.info("Training on documentation code snippets")
                finetune.train_self_supervised_documents(
                    model, tokenizer, train_code, config.train,
                    output_dir=model_config.save_model_dir, report_to='none',
                    train_methods=train_methods,
                )
                
        # Generate practice problems and ideal solutions
        if require_problems:
            train_problems = []
            for data in lib_data.values():
                train_problems.extend(data['problems'])

            if len(train_problems) == 0:
                logger.info("No documentation problems found for training, skipping.")
            else:
                # Can later also add a 'code' field to provide starting code
                train_problems = {'instruction': train_problems}

                if config.train.train_on_practice_problems:
                    logger.info("Training on practice problems and generating solutions")
                    solutions = train_multi_step_sft_with_verification(
                        model, tokenizer, train_problems, config.train,
                        output_dir=model_config.save_model_dir, report_to='none')
                    
                elif config.train.train_on_verified_solutions:
                    logger.info("Generating solutions to practice problems")
                    solutions = generate_solutions(
                        model, tokenizer, train_problems, config.train)
                
                # Train on the verified solutions
                if config.train.train_on_verified_solutions:
                    # Some solutions will be None if no solution was found
                    solutions = [x for x in solutions if x]
                    finetune.train_self_supervised_documents(
                        model, tokenizer, solutions, config.train,
                        output_dir=model_config.save_model_dir, report_to='none',
                        train_methods=train_methods,
                    )
    
    return model