import json
import re
import logging
from datetime import datetime
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional

import scrapy
import html2text
from scrapy.crawler import CrawlerProcess

from src.crawler.utils.docs_finder import find_doc_first_page
from src.crawler.utils.html_processors import extract_main_text_from_html

html2text.config.MARK_CODE = False
LATEST_DOCS_KEYWORDS = ['latest', 'stable', 'current', 'master', 'release', 'main', 'default', ]
EXCLUDE_DOCS_KEYWORDS = ['/version']


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

    def get_allowed_paths(self, url):
        """If the URL is from GitHub, return the path for the specific project."""
        if url.startswith('https://github.com'):
            # First check if the path has the format /user/repo
            path = urlparse(url).path
            if path.count('/') == 2:
                return [path]
            else:
                logging.warning(f"Starting GitHub URL {url} is not in the expected format /user/repo")

        return []

    def parse(self, response):
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
            if link and not link.startswith('http'):
                if not self.allowed_paths or any(link.startswith(path) for path in self.allowed_paths):
                    link = response.urljoin(link)
                    yield scrapy.Request(link, callback=self.parse)


def _format_properly(spider_output: list[dict]) -> dict:
    result_dict = {'content': [], 'code': []}
    for item in spider_output:
        result_dict['content'].append(item['content'])
        result_dict['code'].extend(item['code_blocks'])
    return result_dict


def _get_doc_data_by_url(doc_url: str, file_path: Path = None) -> dict:
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
    # generate a unique id based on the date and a unique identifier
    dir_name = Path('scrapped_docs')
    if not dir_name.exists():
        dir_name.mkdir()
    if not file_path:
        file_path = dir_name/f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
        'FEED_FORMAT': 'json', 
        'FEED_URI': file_path,
        'LOG_LEVEL': 'ERROR'
    })
    process.crawl(DocsSpider, start_urls=[doc_url])
    process.start()

    return _format_properly(json.load(open(file_path)))


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
    return _get_doc_data_by_url(url)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('trafilatura')
    logger.setLevel(logging.INFO)
    print(get_doc_data(scrapy))