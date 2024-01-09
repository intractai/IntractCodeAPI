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

UNALLOWED_FILE_TYPES = set([
    '.jpg', '.jpeg', '.png', '.lib', '.dll', '.bin', '.exe', '.so', '.a',
    '.o', '.out', '.obj', '.pyc', '.class', '.jar', '.war', '.ear', '.zip',
    '.tar', '.gz', '.tgz', '.rar', '.7z', '.pdf', '.doc', '.docx', '.ppt',
    '.pptx', '.webp', '.stl', '.vcxproj', '.sln', '.vcxproj.filters',
    '.svg', '.ico', '.icns', '.ttf', '.woff', '.woff2', '.eot', '.mp3',
    '.mp4', '.wav', '.mov', '.avi', '.mpg', '.mpeg', '.flv', '.wmv',
    '.DS_Store', '.gitmodules', '.gitattributes',
])


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
    parser.add_argument('--max_repo_size', type=int, default=1000,
                        help='Maximum size of repository in kilobytes.')
    parser.add_argument('--min_true_size', type=int, default=10,
                        help='Minimum size of repository in kilobytes post processing.')
    parser.add_argument('--min_true_files', type=int, default=5,
                        help='Minimum number of files after post processing.')
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
        repo_url: str, output_dir: str, auth_token: str, timeout: int = 30,
        max_file_size_kb: int = 30, repo_name: Optional[str] = None,
        min_true_size: Optional[int] = None, min_true_files: Optional[int] = None):
    """Downloads a repository from GitHub and saves it to the specified directory.

    Args:
        repo_url: URL of the repository to download.
        output_dir: Directory to save the downloaded repository.
        auth_token: GitHub personal access token.
        timeout: Timeout in seconds.
        max_file_size_kb: Maximum file size in kilobytes.
        repo_name: Name of the repository. If not specified, will be inferred from `repo_url`.
        min_true_size: Minimum size of repository in kilobytes post processing.
        min_true_files: Minimum number of files after post processing.
    """
    headers = {'Authorization': f'token {auth_token}'}
    response = requests.get(
      repo_url, headers=headers, stream=True, timeout=timeout)
    response.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(response.content))

    # Delete any files that are too large or not allowed
    # Recursively go through all files
    for file in z.filelist:
        if file.file_size > max_file_size_kb * 1024:
            z.filelist.remove(file)
        elif os.path.splitext(file.filename)[1].lower() in UNALLOWED_FILE_TYPES:
            z.filelist.remove(file)

    # Save to output directory under project name
    repo_name = repo_name or os.path.splitext(os.path.basename(repo_url))[0]

    # Check for min number of files and min size
    too_few_files = min_true_files and len(z.filelist) < min_true_files
    too_small_size = min_true_size and \
        sum([file.file_size for file in z.filelist]) < min_true_size * 1024

    if too_few_files or too_small_size:
        print(f'Skipping {repo_name} due to too few files or too small size.')
        return

    # Save to output directory under project name
    z.extractall(output_dir)

    # Rename root folder to project name
    root_folder = z.filelist[0].filename.split('/')[0]
    os.rename(
        os.path.join(output_dir, root_folder),
        os.path.join(output_dir, repo_name))

    # Go back through the new files and delete any unallowed file types
    for root, dirs, files in os.walk(os.path.join(output_dir, repo_name)):
        for file in files:
            if os.path.splitext(file)[1].lower() in UNALLOWED_FILE_TYPES:
                os.remove(os.path.join(root, file))


def save_repos(repos_data: List[Dict[str, Any]], output_dir: str, auth_token: str, **kwargs):
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
        download_and_save_repo(
            repo_url, output_dir, auth_token, repo_name=repo_name, **kwargs)


if __name__ == '__main__':
    args = parse_args()

    # Clean the data directory
    if args.clean:
        if os.path.isdir(PRIMARY_DATA_DIR):
            shutil.rmtree(PRIMARY_DATA_DIR)
        if os.path.isdir(OOD_DATA_DIR):
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
        max_size_kb=args.max_repo_size,
        min_stars=args.min_stars,
        min_date=args.min_date,
        languages=[args.main_lang]
    )

    ood_data = find_gh_repos(
        auth_token=pat,
        n_repos=args.n_repos,
        max_size_kb=args.max_repo_size,
        min_stars=args.min_stars,
        min_date=args.min_date,
        languages=args.ood_langs
    )

    # Save the repositories
    save_repos(
        primary_data, PRIMARY_DATA_DIR, pat, max_file_size_kb=args.max_file_size,
        min_true_size=args.min_true_size, min_true_files=args.min_true_files)
    save_repos(
        ood_data, OOD_DATA_DIR, pat, max_file_size_kb=args.max_file_size,
        min_true_size=args.min_true_size, min_true_files=args.min_true_files)
