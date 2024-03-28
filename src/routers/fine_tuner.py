from typing import Annotated, Dict, List, Optional
from argparse import Namespace
import logging
import threading

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, CancelledError

from src import config_handler, modeling
from src.training import finetune
from src.training.interactive.train_multi_step_sft import (
    generate_solutions,
    train_multi_step_sft_with_verification,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class ProjectFinetuneData(BaseModel):
    project_dict: Optional[Dict[str, str]] = None
    language: Optional[str] = None
    libraries: Optional[List[str]] = None


def verify_request_data(item: ProjectFinetuneData):
    if not (item.project_dict or item.libraries):
        raise ValueError("Either project_dict or libraries must be provided.")
    elif item.libraries and not item.language:
        raise ValueError("Language must be provided if libraries are provided.")


@router.post("/finetune/project")
def finetune_project(item: ProjectFinetuneData, config: Annotated[Namespace, Depends(config_handler.get_config)]):
    try:
        verify_request_data(item)
    except ValueError as e:
        return {'result': 'error', 'message': str(e)}

    try: 
        with ThreadPoolExecutor() as executor:
            # Create a new future for the incoming request
            job_thread = executor.submit(finetune_task, item, config)
            # Run the future and return the result
            result = job_thread.result()
    except CancelledError:
        logger.info("Cancelled /generate execution by a new request.")
        return {'result': 'cancelled'}

    return {'result': 'success'}


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
        'text': ['Documentation page 1', 'Ninjax is a general module system for JAX. It gives the user complete and transparent control over updating the state of each module, bringing the flexibility of PyTorch and TensorFlow to JAX. Moreover, Ninjax makes it easy to mix and match modules from different libraries, such as Flax and Haiku.'],
        'code': ['def train(self, x, y):', 'model = MyModel(3, lr=0.01, name="model")', 'fake code'],
    }

    if gen_problems:
        return_data['problems'] = ['Write a function that takes a list of numbers and returns the sum of the numbers.', 'Write a function that takes a list of numbers and returns the average of the numbers.']

    return return_data


def finetune_task(item: ProjectFinetuneData, config: Namespace):
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
        item (ProjectFinetuneData): The project, language, and library data.
        config (Namespace): The global config.
    """
    item.libraries = item.libraries or []

    # Print the current thread id to show that it is different for each request
    modeling.GLOBAL_FINETUNE_THREAD_ID = threading.get_ident()
    model_config = config.model

    model_provider = modeling.ModelProvider.get_instance()
    tokenizer = model_provider.get_model_utils()['tokenizer']
    model = model_provider.get_model()

    # Determine what data to collect / generate
    require_problems = config.train.train_on_practice_problems \
        or config.train.train_on_verified_solutions
    require_docs = (
        config.train.train_on_docs \
        or config.train.train_on_doc_code \
        or require_problems
    ) and bool(item.libraries)
        
    
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
                get_documentation_data, library, require_problems)

    # Train on the user's codebase while asynchronously collecting documentation
    if config.train.train_on_code and item.project_dict:
        logger.info("Training on user's codebase")
        finetune.train_self_supervised_project(
            model, tokenizer, item.project_dict, config.train,
            output_dir=model_config.save_model_dir, report_to='none',
            train_methods=train_methods, # TODO: Test train_methods
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
                train_documents.extend(data['text'])

            logger.info(f"Training on {len(train_documents)} documents of documentation text")
            finetune.train_self_supervised_documents( # TODO: Test train_self_supervised_documents
                model, tokenizer, train_documents, config.train,
                output_dir=model_config.save_model_dir, report_to='none',
                train_methods=['ntp'],
            )
                
        # Train on the documentation code snippets

        if config.train.train_on_doc_code:
            train_code = []
            for data in lib_data.values():
                train_code.extend(data['code'])

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

            # Can later also add a 'code' field to provide starting code
            train_problems = {'instruction': train_problems}

            if config.train.train_on_practice_problems:
                logger.info("Training on practice problems and generating solutions")
                solutions = train_multi_step_sft_with_verification( # TODO: Test train_sft_with_verification, make sure returned solutions are correct
                    model, tokenizer, train_problems, config.train,
                    output_dir=model_config.save_model_dir, report_to='none')
                
            elif config.train.train_on_verified_solutions:
                logger.info("Generating solutions to practice problems")
                solutions = generate_solutions( # TODO: Test generate_solutions
                    model, tokenizer, train_problems, config.train)
            
            # Train on the verified solutions
            if config.train.train_on_verified_solutions:
                finetune.train_self_supervised_documents(
                    model, tokenizer, solutions, config.train,
                    output_dir=model_config.save_model_dir, report_to='none',
                    train_methods=train_methods,
                )


    return {"result": "success"}
