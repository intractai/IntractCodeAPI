import argparse
import os
from pathlib import Path
import requests


def send_empty_request(port=8000):
    response = requests.post(
        f"http://localhost:{port}/finetune/project",
        json={},
        timeout=10*60,
    )

    output_data = response.json()
    print(f"Response: {output_data}")

def send_codebase_request(port=8000):
    # Make a dictionary of the relative paths to all files
    # in the same directory as this script, and map to the file contents.
    project_dict = {}
    for path in Path(__file__).parent.parent.glob("client_test/*.py"):
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                project_dict[path.name] = f.read()

    response = requests.post(
        f"http://localhost:{port}/finetune/project",
        json={'project_dict': project_dict},
        timeout=10*60,
    )

    output_data = response.json()
    print(f"Response: {output_data}")


def send_library_request(port=8000):
    response = requests.post(
        f"http://localhost:{port}/finetune/project",
        json={'language': 'python', 'libraries': ['ninjax', 'SymPy']},
        timeout=10*60,
    )

    output_data = response.json()
    print(f"Response: {output_data}")


def send_combined_request(port=8000):
    project_dict = {}
    for path in Path(__file__).parent.parent.glob("client_test/*.py"):
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                project_dict[path.name] = f.read()

    response = requests.post(
        f"http://localhost:{port}/finetune/project",
        json = {
            'project_dict': project_dict,
            'language': 'python',
            'libraries': ['ninjax', 'SymPy']
        },
        timeout = 10 * 60,
    )

    output_data = response.json()
    print(f"Response: {output_data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send a request to the API.')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API.')
    args = parser.parse_args()

    # send_empty_request(args.port)
    # send_codebase_request(args.port)
    # send_library_request(args.port)
    send_combined_request(args.port)
