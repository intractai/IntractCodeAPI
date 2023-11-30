import argparse
import os
from pathlib import Path
import requests


def send_request(port=8000):
    # Make a dictionary of the relative paths to all files
    # in the same directory as this script, and map to the file contents.
    project_dict = {}
    for path in Path(__file__).parent.glob("*"):
        if path.is_file():
            with open(path, "r") as f:
                project_dict[path.name] = f.read()

    response = requests.post(f"http://localhost:{port}/finetune/project", json={"project_dict": project_dict})

    output_data = response.json()
    print(f"Response: {output_data}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send a request to the API.')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API.')
    args = parser.parse_args()
    send_request(args.port)