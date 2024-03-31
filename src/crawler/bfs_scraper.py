import requests
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from src.crawler.utils.html_processors import extract_main_text_from_html
from src.crawler.utils.docs_finder import find_doc_first_page


def _bfs_scrapper(start_url: str, max_char_count: int = 10000) -> str:
    """
    start_url: str
    max_char_count: int
    """
    queue = [start_url]
    visited = set()
    content = ''

    allowed_domains = [urlparse(start_url).netloc]

    while queue:
        current_url = queue.pop(0)
        if current_url in visited:
            continue

        # Check if the URL is in the allowed domains
        url_domain = urlparse(current_url).netloc
        if url_domain not in allowed_domains:
            continue

        logging.info(f"Visiting: {current_url}")
        logging.info(f"Char number progress: {len(content)}/{max_char_count}")
        visited.add(current_url)

        # Fetch the content of the URL
        try:
            response = requests.get(current_url)
            if response.status_code != 200:
                continue
            
            page_content = extract_main_text_from_html(response.content.decode('utf-8'))
            soup = BeautifulSoup(response.text, 'lxml')

            content += page_content + '\n\n'
            if len(content) >= max_char_count:
                return content

            # Find all links and add them to the queue if not visited
            for link in soup.find_all('a', href=True):
                absolute_link = urljoin(current_url, link['href'])
                if absolute_link not in visited:
                    queue.append(absolute_link)

        except Exception as e:
            print(f"Failed to fetch {current_url}: {str(e)}")
        
        # Validating the content length
    if len(content) < 5:
        raise ValueError('Content is too short')
        
    return content


def bfs_scrapper(library: str, max_char_count: int = 10000) -> str:
    """
    library: str
    max_char_count: int
    """
    url = find_doc_first_page(library)
    return _bfs_scrapper(url, max_char_count)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(bfs_scrapper('scrapy'))


