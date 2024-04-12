import multiprocessing
from typing import Generator, Optional
import json
import re
import logging
import tempfile
import os
import time
import requests
from pathlib import Path
from queue import Queue
from abc import ABC, abstractmethod

import scrapy
import html2text
import git
from scrapy.crawler import CrawlerProcess
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

from src.crawler.utils.docs_finder import find_doc_first_page
from src.crawler.utils.html_processors import extract_main_text_from_html

html2text.config.MARK_CODE = False
# Not using word "latest" because some projects will have a link that says "to the latest version"
LATEST_DOCS_KEYWORDS = ['stable', 'current', 'master', 'release', 'main', 'default', ]
EXCLUDE_DOCS_KEYWORDS = ['/version']
DOC_EXTENSIONS_ON_GITHUB = ('.md', '.rst', '.txt', '.html')
MAX_OVERVIEW_TOKENS = 10000

#TODO: all languages are not supported; we also need to think about what 
# we're going to do with text files, json files, and so on when we're dealing with a repo
LANG_TO_EXT = { 
    'python': '.py',
    'javascript': '.js',
    'typescript': '.ts',
    'java': '.java',
    'c': '.c',
    'c++': '.cpp',
    'c#': '.cs',
    'go': '.go',
    'php': '.php',
    'ruby': '.rb',
    'rust': '.rs',
    'kotlin': '.kt',
    'swift': '.swift',
    'dart': '.dart',
    'r': '.r',
    'scala': '.scala',
    'shell': '.sh',
    'bash': '.sh',
    'powershell': '.ps1',
    'perl': '.pl',
    'lua': '.lua',
    'html': '.html',
    'css': '.css',
    'scss': '.scss',
    'sass': '.sass',
    'less': '.less',
    'stylus': '.styl',
    'sql': '.sql',
    'dockerfile': 'Dockerfile',
}


logger = logging.getLogger(__name__)


def _delete_file_after_1_hour(temp_file_path: Path) -> None:
    time.sleep(3600)  # Wait for 1 hour
    if temp_file_path.exists():
        os.remove(temp_file_path)


class DocsSpider(scrapy.Spider):
    name = 'docs'

    def __init__(self, start_urls: list, *args, **kwargs):
        super(DocsSpider, self).__init__(*args, **kwargs)
        if len(start_urls) != 1:
            raise ValueError('Exactly one URL is allowed!')

        self.start_urls = start_urls
        self.allowed_domains = [urlparse(url).netloc for url in start_urls]
        self.allowed_paths = set(self.get_allowed_paths(start_urls[0]))
        self.h = html2text.HTML2Text()
        self.h.ignore_links = True

    def get_allowed_paths(self, url: str):
        """If the URL is from GitHub, return the path for the specific project."""
        if url.startswith('https://github.com'):
            # First check if the path has the format /user/repo
            path = urlparse(url).path
            if path.count('/') == 2:
                return [path]
            else:
                logging.warning(f"Starting GitHub URL {url} is not in the expected format /user/repo")

        return []

    def parse(self, response: scrapy.http.Response) -> Generator[dict, None, None]:
        try:
            if 'text/html' not in response.headers.get('Content-Type').decode('utf-8'):
                return
            logging.info(f"Extracting content from: {response.url}")
            content = extract_main_text_from_html(response.body.decode('utf-8'))
            is_latest_doc = any(keyword in response.url for keyword in LATEST_DOCS_KEYWORDS)
            has_version = re.search(r'/\d+(\.\d+)+/', response.url)
            is_old_doc = any(keyword in response.url for keyword in EXCLUDE_DOCS_KEYWORDS)

            # This seems like a naive way to extract code blocks from the HTML content
            # but as far as I looked other libraries were doing the same thing
            # we can do more complex things but that's TODO
            code_blocks = []
            for code_block in response.xpath('//pre'):
                code_block = self.h.handle(code_block.get())
                code_blocks.append(code_block)

            if is_latest_doc or (not has_version and not is_old_doc): 
                yield {
                    'content': content,
                    'code_blocks': code_blocks,
                }
            else:
                return
                
        except Exception as e:
            logging.info(f"Failed to extract content from {response.url}: {str(e)}")

        # To follow links to next pages and continue crawling, but only within the latest documentation
        try:
            links = response.css('a::attr(href)').getall()
        except Exception as e:
            logging.info(f"Failed to extract links from {response.url}: {str(e)}")
            links = []
            
        for link in links:
            # check if LATEST_DOCS_KEYWORDS are in the link
            if link and not link.startswith('http') and not link.startswith('mailto:'):
                if not self.allowed_paths or any(link.startswith(path) for path in self.allowed_paths):
                    link = response.urljoin(link)
                    yield scrapy.Request(link, callback=self.parse)


class Scraper(ABC):

    def __init__(self, start_urls: list, file_path: Path = None):
        self._start_urls = start_urls
        self._file_path = self._create_file_path(file_path)

    def _create_file_path(self, file_path: Path):
        if file_path is None:
            file_path = Path(tempfile.mktemp().replace('\\', '/'))
            # threading.Thread(target=_delete_file_after_1_hour, args=(file_path,)).start()
            
        # check if the file path is a directory
        assert not file_path.is_dir(), "The file path should not be a directory."

        # create the directory to file path if it does not exist
        if not file_path.parent.exists():
            logger.info(f"Creating directory: {file_path.parent}")
            file_path.parent.mkdir()

        return Path(file_path)
    
    def _save_to_db(self, data: dict):
        """Saving the documentation into the database

        Args:
            data (dict): the output dictionary is in this format:
            {
                'content': [list of all the content extracted from the documentation],
                'code': [list of all the code blocks extracted from the documentation]
            }
        """
        pass #TODO: to be implemented in the future
    
    @abstractmethod
    def scrape(self) -> dict:
        pass


class GithubScraper(Scraper):
    
    def _clone_repo(self, repo_url: str, target_path: str) -> bool:
        try:
            git.Repo.clone_from(repo_url, target_path)
            return True
        except Exception as e:
            logger.warning(f"Error cloning {repo_url}: {e}")
            return False

    def scrape(self, limit_tokens: bool = False) -> dict:
        """
        Use BFS to search through the given github repository and 
        extract the content and code blocks from the documentation
        """
        repo_url = self._start_urls[0] #TODO: this is just temporary until I implement the logic to do it for multiple repos
        target_dir = self._file_path
        self._clone_repo(repo_url, target_dir)
        docs = {
            'content': [],
            'code': []
        }
        docs_token_count = 0
        queue = Queue()
        root_path = Path(target_dir)

        queue.put(root_path)

        while not queue.empty():
            current_path = queue.get()
            for entry in current_path.iterdir():
                if entry.is_dir():
                    queue.put(entry)
                # TODO: need to add ability to extract code blocks from the documentation files
                elif entry.suffix in DOC_EXTENSIONS_ON_GITHUB:
                    try:
                        content = entry.read_text(encoding='utf-8')
                        docs['content'].append(content)
                        docs_token_count += len(content)
                        if limit_tokens and docs_token_count >= MAX_OVERVIEW_TOKENS:
                            return {'overview': '\n\n'.join(docs['content'])[:MAX_OVERVIEW_TOKENS]}
                    except Exception as e:
                        print(f"Error reading {entry}: {e}")
                # TODO: limit_tokens is a nasty hack; need to refactor this later
                elif entry.suffix in LANG_TO_EXT.values() and not limit_tokens:
                    try:
                        content = entry.read_text(encoding='utf-8')
                        docs['code'].append(content)
                    except Exception as e:
                        print(f"Error reading {entry}: {e}")

        self._save_to_db(docs)

        if limit_tokens:
            if docs_token_count <= 10:
                raise ValueError('Content is too short')
            return {'overview': '\n\n'.join(docs['content'])}

        return docs


class AsyncDocsScraper(Scraper):

    def _format_properly(self, spider_output: list[dict]) -> dict:
        result_dict = {'content': [], 'code': []}
        for item in spider_output:
            result_dict['content'].append(item['content'])
            result_dict['code'].extend(item['code_blocks'])
        return result_dict
    
    def run_crawler(self, start_urls: list, file_path: Path):
        process = CrawlerProcess({
            'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
            'FEED_FORMAT': 'json',
            'FEED_URI': file_path,
            'LOG_LEVEL': 'ERROR',
            'COMPRESSION_ENABLED': 'False',
        })
        process.crawl(DocsSpider, start_urls=start_urls)
        process.start(install_signal_handlers=False)

    def scrape(self):
        """extracts the content and code blocks from the documentation

        Args:
            doc_url (str): _description_

        Returns:
            dict: the output dictionary is in this format:
            {
                'content': [list of all the content extracted from the documentation],
                'code': [list of all the code blocks extracted from the documentation]
            }
        """
        file_path = self._file_path
        queue = multiprocessing.Queue()

        # Make this happen in a different process because it can only be done once
        # per process as part of the way it's designed.
        process = multiprocessing.Process(
            target=self.run_crawler, args=(queue, self._start_urls, file_path))
        process.start()
        process.join()

        with open(file_path, 'r') as f:
            data = json.load(f)

        self._save_to_db(data)

        return self._format_properly(data)
    

class SyncDocsScraper(Scraper):

    def scrape(self, limit_tokens: bool = False):
        """extracts the content and code blocks from the documentation

        Returns:
            dict: the output dictionary is in this format:
            {
                'overview': [list of all the content extracted from the documentation],
            }
        """
        start_url = self._start_urls[0] #TODO: this is just temporary until I implement the logic to do it for multiple repos
        queue = [start_url]
        visited = set()
        content = []
        char_count = 0

        allowed_domains = [urlparse(start_url).netloc]

        while queue:
            current_url = queue.pop(0)
            if current_url in visited:
                continue

            # Check if the URL is in the allowed domains
            url_domain = urlparse(current_url).netloc
            if url_domain not in allowed_domains:
                continue

            logging.debug(f"Visiting: {current_url}")
            print(f"Visiting: {current_url}")
            if limit_tokens:
                logging.debug(f"Char number progress: {char_count}/{MAX_OVERVIEW_TOKENS}")
                print(f"Char number progress: {char_count}/{MAX_OVERVIEW_TOKENS}")
            visited.add(current_url)

            # Fetch the content of the URL
            try:
                response = requests.get(current_url)
                if response.status_code != 200:
                    continue
                
                page_content = extract_main_text_from_html(response.content.decode('utf-8'))
                char_count += len(page_content)
                soup = BeautifulSoup(response.text, 'lxml')

                content.append(page_content)
                if char_count >= MAX_OVERVIEW_TOKENS and limit_tokens:
                    return '\n\n'.join(content)

                # Find all links and add them to the queue if not visited
                for link in soup.find_all('a', href=True):
                    absolute_link = urljoin(current_url, link['href'])
                    if absolute_link not in visited and not link['href'].startswith('mailto:'):
                        queue.append(absolute_link)

            except Exception as e:
                print(f"Failed to fetch {current_url}: {str(e)}")
            
        # Validating the content length
        if char_count < 5:
            raise ValueError('Content is too short')
            
        if limit_tokens:
            return '\n\n'.join(content)
        return content


def get_doc_data(library: str, language: Optional[str]) -> dict:
    """extracts the content and code blocks from the documentation given library name

    Args:
        library (str): name of the library

    Returns:
        dict: the output dictionary is in this format:
        {
            'content': [list of all the content extracted from the documentation],
            'code': [list of all the code blocks extracted from the documentation]
        }
    """
    url = find_doc_first_page(library, language)
    print(f"[get_docs_data] Found documentation URL: {url}")

    if '//github.com/' in url:
        return GithubScraper([url]).scrape()
    return AsyncDocsScraper([url]).scrape()
    

def get_docs_overview(library: str, language: Optional[str]) -> str:
    """extracts the content from the documentation given library name

    Args:
        library (str): name of the library
        max_tokens (int): maximum number of tokens to extract

    Returns:
        str: the content extracted from the documentation
    """
    url = find_doc_first_page(library, language)
    print(f"[get_docs_overview] Found documentation URL: {url}")

    if '//github.com/' in url:
        return GithubScraper([url]).scrape(limit_tokens=True)
    return SyncDocsScraper([url]).scrape(limit_tokens=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('trafilatura')
    logger.setLevel(logging.INFO)
    print(get_doc_data(scrapy))