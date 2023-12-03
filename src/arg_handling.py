import argparse
from argparse import Namespace
import os
from typing import Dict, List, Optional, Tuple, Union


def implicit_convert_string(string: str) -> Union[str, bool, int, float]:
    """Convert a string to a bool, int, or float if possible."""

    # Check for boolean
    if string.lower() in ["true", "false"]:
        return string.lower() == "true"

    # Check if string is numeric (integer or float)
    if string.replace(".", "", 1).isdigit():
        # Differentiate between integer and float
        return int(string) if "." not in string else float(string)

    # Return as string if no conversion is possible
    return string


def parse_args(env_prefixes: Optional[List] = None) -> Tuple[Namespace, Dict[str, Namespace], List]:
    """Parse the arguments for the API and standalone runs."""

    parser = argparse.ArgumentParser(description="Run the API.")
    parser.add_argument("--model_dir", type=str, default="microsoft/codebert-base",
                        help="Model name or path to model")
    parser.add_argument("--local_model_dir", type=str, default="./.download",
                        help="Path to store the model")
    parser.add_argument("--fp16", action="store_true",
                        help="Run with fp16 intead of fp32.")
    parser.add_argument("--context_length", type=int, default=512,
                        help="Maximum context length")
    parser.set_defaults(fp16=False)

    known_args, unknown_args = parser.parse_known_args()

    # Add environemnt variables if relevant
    env_args = {}
    if env_prefixes:
        for prefix in env_prefixes:
            env_vars = {
                k: implicit_convert_string(v) for k, v in os.environ.items()
                if k.startswith(prefix)
            }

            # Remove the prefix from the env vars, and lowercase them, and create a dict
            env_vars = {
                k.replace(prefix, "").lower(): v
                for k, v in env_vars.items()
            }

            env_args[prefix.lower().rstrip("_")] = Namespace(**env_vars)

            # # Combine env_vars into kown_args with precedence for env_vars
            # known_args = Namespace(**{**vars(known_args), **env_vars})

    return known_args, env_args, unknown_args
