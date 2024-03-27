from datasets import Dataset

from src.training.finetune import (
    prepare_raw_project_dataset,
    prepare_raw_document_dataset,
)


def test_prepare_raw_project_dataset():
    project_data = {
        "path/to/file1.py": "print('Hello, world!')",
        "path/to/file2.py": "def foo(): pass"
    }
    expected_output = [{'file_path': 'path/to/file1.py', 'code': "print('Hello, world!')"}, {'file_path': 'path/to/file2.py', 'code': "def foo(): pass"}]
    
    dataset = prepare_raw_project_dataset(project_data)
    
    assert isinstance(dataset, Dataset)
    assert dataset.num_rows == len(project_data)
    assert dataset[0] == expected_output[0]
    assert dataset[1] == expected_output[1]

def test_prepare_raw_project_dataset_empty():
    project_data = {}
    
    dataset = prepare_raw_project_dataset(project_data)
    
    assert isinstance(dataset, Dataset)
    assert dataset.num_rows == 0

def test_prepare_raw_document_dataset():
    documents = ["Document 1 content", "Document 2 content"]
    expected_output = [{'text': "Document 1 content"}, {'text': "Document 2 content"}]
    
    dataset = prepare_raw_document_dataset(documents)
    
    assert isinstance(dataset, Dataset)
    assert dataset.num_rows == len(documents)
    assert dataset[0] == expected_output[0]
    assert dataset[1] == expected_output[1]

def test_prepare_raw_document_dataset_empty():
    documents = []
    
    dataset = prepare_raw_document_dataset(documents)
    
    assert isinstance(dataset, Dataset)
    assert dataset.num_rows == 0
