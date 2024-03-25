import argparse
import os
from functools import partial

from datasets import Dataset, load_dataset


SAVE_DIR = 'data'


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--language', type=str, default=None)
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('-n', type=int, default=None)
    parser.add_argument('--stream', action='store_true', default=False)

    return parser.parse_args()


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


if __name__ == '__main__':
    # Get args
    args = parse_args()

    # Load the dataset
    split = args.split + f'[:{args.n}]' if args.n and not args.stream else args.split
    dataset = load_dataset(
        'bigcode/starcoderdata',
        data_dir = args.language,
        split = split,
        cache_dir = f'{SAVE_DIR}/cache',
        streaming = args.stream
    )

    # Prepare the save directory
    save_path = os.path.join(SAVE_DIR, args.language or 'all', args.split)
    if args.n:
        save_path += f'-{args.n}'

    # Save to local data folder
    if args.stream:
        # Transform iterable dataset into a regular dataset
        partial_dataset = dataset.take(args.n or len(dataset))
        partial_dataset = Dataset.from_generator(
            partial(gen_from_iterable_dataset, partial_dataset),
            features=partial_dataset.features)

        if os.path.exists(save_path):
            os.remove(save_path)

        # Save the dataset
        partial_dataset.save_to_disk(save_path)
    else:
        dataset.save_to_disk(SAVE_DIR)
