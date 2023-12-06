import os
from pathlib import Path
import sys

sys.path.append('./')
from src import finetune, modeling
from src.arg_handling import parse_args


def train(args, finetune_args):
    """Finetune a model with the given arguments."""
    project_dict = {}
    for path in Path(__file__).parent.glob("test_files/*"):
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                project_dict[path.name] = f.read()
    modeling.intialize_model(args.model_dir, args.local_model_dir, args)
    finetune.train_supervised_projectdir(
        project_dict,
        output_dir=args.local_model_dir,
        report_to='none',
        **vars(finetune_args),
    )


if __name__ == "__main__":
    args, env_args, unknown = parse_args(env_prefixes=["FINETUNE_"])
    train(args, env_args['finetune'])
