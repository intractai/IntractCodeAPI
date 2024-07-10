import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union


# Relative to this file
DATA_DIR = 'data/'
METADATA_FILENAME = 'metadata.json'
TASK_TRAIN_DIR = 'train/'
TASK_TEST_DIR = 'test/'
TASK_CODE_DIR = 'code/'
TASK_DOCS_DIR = 'documents/'
TASK_LINKS_FILENAME = 'links.txt'


logger = logging.Logger(__name__)


def combine_project_data(multi_project_data: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Combine data from multiple projects into a single dictionary.

    Args:
        multi_project_data: Dictionary mapping project names to dictionaries mapping
            file names to file contents.

    Returns:
        Dictionary mapping file names to file contents.
    """

    all_data = {}
    for project_name, project_data in multi_project_data.items():
        for file_path, file_data in project_data.items():
            all_data[os.path.join(project_name, file_path)] = file_data

    return all_data


def load_projects(
        relative_path: str, combine_projects: bool = True,
    ) -> Union[Dict[str, Dict[str, str]], Dict[str, str]]:
    """Load project data from a directory.

    Args:
        relative_path: Path to directory containing projects.
        combine_projects: Whether to combine data from multiple projects into a single, flat dictionary.

    Returns:
        Dictionary mapping project names to dictionaries mapping file names to file contents.
        If `combine_projects` is True, returns a single dictionary mapping file names to file contents.
    """
    all_project_data = {}
    data_dir = Path(__file__).parent / relative_path
    # Loop through each project in this directory
    for i, project_path in enumerate(data_dir.glob('*')):
        if project_path.is_dir():
            project_data = {}
            # Loop through each file in this project
            for file_path in project_path.rglob('*'):
                # NOTE: DON'T COMMIT THIS!!!!!
                if file_path.is_file() and file_path.suffix == '.py':
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_text = f.read()
                        if len(file_text) > 0:
                            # Path to file starting from the name of the project folder
                            rel_path = file_path.relative_to(project_path.parent)
                            project_data[str(rel_path)] = file_text
                    except UnicodeDecodeError:
                        logging.warning(f'Error reading file {file_path}. Skipping...')
            all_project_data[project_path.name] = project_data

    # if combine_projects:
    #     all_project_data = combine_project_data(all_project_data)

    if combine_projects:
        combined_project_data = {}
        for project_data in all_project_data.values():
            combine_project_data = {**combined_project_data, **project_data}
        all_project_data = combine_project_data

    return all_project_data


def load_documents(relative_path: str) -> List[str]:
    """Load documents from a directory.

    Args:
        relative_path: Path to directory containing documents.

    Returns:
        List of document texts.
    """
    all_documents = []
    data_dir = Path(__file__).parent / relative_path
    for doc_path in data_dir.glob('**/*'):
        if doc_path.is_file():
            with open(doc_path, 'rb') as f:
                doc_bytes = f.read()
            if len(doc_bytes) > 0:
                all_documents.append((doc_path.name, doc_bytes))
                logger.debug(f'Loaded document: {doc_path}')

    return all_documents


def load_links(relative_path: str) -> List[str]:
    """Load links from a file.

    Args:
        relative_path: Path to file containing links.

    Returns:
        List of links.
    """
    all_links = []
    data_dir = Path(__file__).parent / relative_path
    with open(data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) > 0:
                all_links.append(line.strip())

    return all_links


def load_task_info() -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Load all the info (code, documents, and links) for a single eval task.
    Args:
        relative_path: Path to file containing task information.

    Returns:
        Tuple of task name and dictionary containing task information.
        The task info dictionary contains the following keys:
            - 'train': Dictionary containing the following keys:
                - 'code' (Dict[str, str]): Dictionary mapping file names to file contents.
                - 'docs' (List[Tuple[str, bytes]]): List of document names and byte encodings.
                - 'links' (List[str]): List of links.
            - 'test' (Dict[str, str]): Dictionary mapping project names to dictionaries mapping file names to file contents.
            - 'metadata' (Dict[str, str]): Dictionary containing metadata about the task.
    """
    data_dir = Path(__file__).parent / DATA_DIR
    # Get all folders in the directory (1 folder = 1 task)
    for task_path in data_dir.glob('*'):
        if task_path.is_dir():
            task_info = {'train': {}, 'test': {}, 'metadata': {}}
            task_name = task_path.name

            # Paths to relevant files / directories
            metadata_path = task_path / METADATA_FILENAME
            train_path = task_path / TASK_TRAIN_DIR
            code_path = train_path / TASK_CODE_DIR
            docs_path = train_path / TASK_DOCS_DIR
            links_path = train_path / TASK_LINKS_FILENAME
            test_path = task_path / TASK_TEST_DIR

            # Load metadata
            if metadata_path.is_file():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                task_info['metadata'] = metadata

            # Load train data
            if code_path.is_dir():
                task_info['train']['code'] = load_projects(code_path)
            if docs_path.is_dir():
                task_info['train']['docs'] = load_documents(docs_path)
            if links_path.is_file():
                task_info['train']['links'] = load_links(links_path)

            # Load test data
            if test_path.is_dir():
                task_info['test'] = load_projects(test_path)

            if len(task_info['train']) == 0:
                logger.warning(f'Task {task_name} has no train data. Skipping...')
                continue
            elif len(task_info['test']) == 0:
                logger.warning(f'Task {task_name} has no test data. Skipping...')
                continue

            yield task_name, task_info
