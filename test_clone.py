from pathlib import Path
from queue import Queue
import logging
import tempfile

import git  # pip install gitpython


logger = logging.getLogger(__name__)

DOC_EXTENSIONS_ON_GITHUB = ('.md', '.rst', '.txt', '.html')


def bfs_find_and_read_docs(root_dir):
    """
    Use BFS to search through the given directory for document files
    (.md, .rst, .txt, .html) and return their paths and contents.
    """
    docs = []
    queue = Queue()
    root_path = Path(root_dir)
    
    # Check if root exists and is a directory
    if not root_path.is_dir():
        print(f"The path {root_dir} does not exist or is not a directory.")
        return docs

    queue.put(root_path)

    while not queue.empty():
        current_path = queue.get()
        for entry in current_path.iterdir():
            if entry.is_dir():
                queue.put(entry)
            elif entry.suffix in DOC_EXTENSIONS_ON_GITHUB:
                try:
                    content = entry.read_text(encoding='utf-8')
                    docs.append(content)
                except Exception as e:
                    print(f"Error reading {entry}: {e}")

    return docs


def clone_repo(repo_url, target_dir):
    try:
        git.Repo.clone_from(repo_url, target_dir)
        return True
    except Exception as e:
        logger.warning(f"Error cloning {repo_url}: {e}")
        return False


if __name__ == "__main__":
    repo_url = "https://github.com/danijar/ninjax"
    target_dir = 'tmp/ninjax'
    with tempfile.TemporaryDirectory() as target_dir:
        print(target_dir)
        # clone_repo(repo_url, target_dir)
        # docs = bfs_find_and_read_docs(target_dir)
    # print(docs)