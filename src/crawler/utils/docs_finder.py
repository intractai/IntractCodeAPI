from googlesearch import search


QUERY_TEMPLATE = "Latest {} library documentation"
lib_to_doc = {
    'html2text': 'https://html2text.readthedocs.io/en/latest/',
    'pytorch': 'https://pytorch.org/docs/stable/package.html#tutorials'
}


def find_doc_first_page(library: str):
    if library in lib_to_doc:
        return lib_to_doc[library]
    query = QUERY_TEMPLATE.format(library)
    url = list(search(query, sleep_interval=0, num_results=1, advanced=False, lang='en'))[0]
    return url