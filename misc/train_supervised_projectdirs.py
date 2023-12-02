import sys
sys.path.append('./')

from src import finetune
from pathlib import Path

import os
import argparse

def train(args):
    env_vars = {k: v for k, v in os.environ.items() if k.startswith("FINETUNE_")}
    #Remove the prefix from the env vars, and lowercase them, and create a dict
    env_vars = {k.replace("FINETUNE_", "").lower(): v for k, v in env_vars.items()}
    project_dict = {}
    for path in Path(__file__).parent.glob("*"):
        if path.is_file():
            with open(path, "r") as f:
                project_dict[path.name] = f.read()
    finetune.train_supervised_projectdir(project_dict,
                                        model_name_or_path=args.model_dir,output_dir=args.model_dir, 
                                        report_to='none',**env_vars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send a request to the API.')
    parser.add_argument('model_dir', type=str, default='microsoft/codebert-base', help='Model name or path to model')
    args=parser.parse_args()
    train(args)