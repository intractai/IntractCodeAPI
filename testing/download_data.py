"""Pulls data from several random repositories.

This script pulls random repositories from GitHub based on a number
of parameters, and then saves them for later use in evaluation.
Pulled repositories are split into two groups, one for finetuning
and one for OOD evaluation (i.e. when training on finetuning subset,
how much does performance on the OOD subset decrease?).
"""

import argparse
import datetime
import io
import os
import random
import shutil
from typing import Any, Dict, List, Optional
import zipfile

import requests


GH_SEARCH_URL = 'https://api.github.com/search/repositories'
GH_DOWNLOAD_URL = 'https://api.github.com/repos'
PRIMARY_DATA_DIR = 'data/finetune/'
OOD_DATA_DIR = 'data/ood/'


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--pat_path', type=str, default='github.pat',
                        help='Path to GitHub PAT.')
    parser.add_argument('--main_lang', type=str, default='python',
                        help='Main language to pull data from.')
    parser.add_argument('--ood_langs', type=str, nargs='+', default=['c++', 'javascript'],
                        help='OOD languages to pull data from.')
    parser.add_argument('--n_repos', type=int, default=10,
                        help='Number of repositories to pull from for finetuning and OOD data.')
    parser.add_argument('--min_stars', type=int, default=40,
                        help='Minimum number of stars for repositories to pull from.')
    # Dynamic default last 60 days
    two_months_ago = datetime.datetime.now() - datetime.timedelta(days=60)
    parser.add_argument('--min_date', type=str, default=two_months_ago.strftime('%Y-%m-%d'),
                        help='Minimum date for repositories to pull from.')
    parser.add_argument('--max_size', type=int, default=5000,
                        help='Maximum size of repository in kilobytes.')
    parser.add_argument('--max_file_size', type=int, default=30,
                        help='Maximum file size of repository in kilobytes.')
    parser.add_argument('--clean', action='store_true', default=False,
                        help='Whether to clean the data directory before downloading.')

    cmd_args = parser.parse_args()
    return cmd_args


def find_gh_repos(
        auth_token: str,
        n_repos: int,
        max_size_kb: int = 10_000,
        min_stars: int = 0,
        min_date: Optional[str] = None,
        languages = Optional[List[str]],
    ) -> List[Dict[str, Any]]:
    """Retrieves repository data from GitHub.
    
    Args:
        auth_token: GitHub personal access token.
        n_repos: Number of repositories to pull.
        max_size_kb: Maximum size of repository in kilobytes.
        min_stars: Minimum number of stars for repositories to pull.
        min_date: Minimum date for repositories to pull.
        languages: List of languages to pull from.
        
    Returns:
        List of dictionaries containing repository data.
    """

    headers = {
        'Authorization': f'token {auth_token}',
        'Accept': 'application/vnd.github.v3+json',
    }

    query = f'size:<={max_size_kb}'

    if min_stars > 0:
        query += f' stars:>={min_stars}'

    if min_date:
        query += f' created:>={min_date}'

    if languages:
        for language in languages:
            query += f' language:{language}'

    per_page = 100
    repo_idx = 0
    page_idx = 1

    repo_data = []
    while repo_idx < n_repos:
        response = requests.get(GH_SEARCH_URL, headers=headers, params={
            'q': query,
            'per_page': per_page,
            'sort': 'stars',
            'order': 'desc',
            'page': page_idx
        }, timeout=10)
        response.raise_for_status()
        page_idx += 1

        data = response.json()

        if repo_idx == 0:
            total_count = data['total_count']
            n_repos = min(total_count, n_repos)

        # Randomize order of repositories
        random.shuffle(data['items'])

        for repo in data['items']:
            repo_data.append({
                'name': repo['name'],
                'owner': repo['owner']['login'],
                'clone_url': repo['clone_url'],
                'stars': repo['stargazers_count'],
                'size_kb': repo['size'],
                'language': repo['language']
            })
            repo_idx += 1

            if repo_idx >= n_repos:
                break

    return repo_data


def download_and_save_repo(
        repo_url: str, output_dir: str, auth_token: str,
        timeout: int = 30, max_file_size_kb: int = 30):
    """Downloads a repository from GitHub and saves it to the specified directory.

    Args:
        repo_url: URL of the repository to download.
        output_dir: Directory to save the downloaded repository.
        auth_token: GitHub personal access token.
        timeout: Timeout in seconds.
        max_file_size_kb: Maximum file size in kilobytes.
    """
    headers = {'Authorization': f'token {auth_token}'}
    response = requests.get(
      repo_url, headers=headers, stream=True, timeout=timeout)
    response.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(response.content))

    # Delete any files that are too large
    # TODO: This could lead to an empty repository, maybe fix this eventually
    for file in z.filelist:
        if file.file_size > max_file_size_kb * 1024:
            z.filelist.remove(file)

    z.extractall(path=output_dir)


def save_repos(repos_data: List[Dict[str, Any]], output_dir: str, auth_token: str):
    """Saves a list of repositories to a specified directory.

    Args:
        repos_data: List of repository data.
        output_dir: Directory to save the repositories.
        auth_token: GitHub personal access token.
    """
    for repo in repos_data:
        owner = repo['owner']
        repo_name = repo['name']
        repo_url = f'{GH_DOWNLOAD_URL}/{owner}/{repo_name}/zipball'
        download_and_save_repo(repo_url, output_dir, auth_token)


if __name__ == '__main__':
    args = parse_args()

    # Clean the data directory
    if args.clean:
        shutil.rmtree(PRIMARY_DATA_DIR)
        shutil.rmtree(OOD_DATA_DIR)

    # Create data directories if they don't exist
    os.makedirs(PRIMARY_DATA_DIR, exist_ok=True)
    os.makedirs(OOD_DATA_DIR, exist_ok=True)

    # Load GitHub personal access token
    assert os.path.isfile(args.pat_path), \
        f"Please create a GitHub personal access token and save it to {args.pat_path}."

    with open(args.pat_path, 'r', encoding='utf-8') as f:
        pat = f.read().strip()

    # Pull data from GitHub
    primary_data = find_gh_repos(
        auth_token=pat,
        n_repos=args.n_repos,
        max_size_kb=args.max_size,
        min_stars=args.min_stars,
        min_date=args.min_date,
        languages=[args.main_lang]
    )

    ood_data = find_gh_repos(
        auth_token=pat,
        n_repos=args.n_repos,
        max_size_kb=args.max_size,
        min_stars=args.min_stars,
        min_date=args.min_date,
        languages=args.ood_langs
    )

    # Save the repositories
    save_repos(primary_data, PRIMARY_DATA_DIR, pat)
    save_repos(ood_data, OOD_DATA_DIR, pat)
