import re
from googlesearch import search


QUERY_TEMPLATE = "{} library documentation"
lib_to_doc = {
    'html2text': 'https://html2text.readthedocs.io/en/latest/',
    'pytorch': 'https://pytorch.org/docs/stable/package.html#tutorials'
}


def _reduce_github_url(url: str) -> str:
    match = re.match(r'(https://github\.com/[^/]+/[^/]+)', url)
    if match:
        return match.group(1)
    else:
        return url  # Return the original URL if it doesn't match the pattern


def _format_url(url: str) -> str:
    if '/github.com/' in url:
        url = _reduce_github_url(url)
    return url


def find_doc_first_page(library: str):
    if library in lib_to_doc:
        return lib_to_doc[library]
    query = QUERY_TEMPLATE.format(library)
    url = list(search(query, sleep_interval=0, num_results=2, advanced=False, lang='en'))[0]

    url = _format_url(url)
    return url