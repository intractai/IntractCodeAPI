import sys
sys.path.append('./')

from src import finetune
from pathlib import Path
from src import model as model_mdl

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
    model_mdl.intialize_model(args.model_dir, args.local_model_dir, args)
    finetune.train_supervised_projectdir(project_dict,
                                         output_dir=args.local_model_dir,
                                        report_to='none',**env_vars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send a request to the API.')
    parser.add_argument('model_dir', type=str, default='microsoft/codebert-base', help='Model name or path to model')
    parser.add_argument('--local_model_dir', type=str, default='./.download', help='Path to store the model')
    parser.add_argument('--fp16', type=bool, default=False, help='Run with fp32 intead of fp16.')
    args=parser.parse_args()
    train(args)