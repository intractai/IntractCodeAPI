import csv
import os
from threading import Thread
import time

from ..docs_scraper import get_doc_data


def load_existing_data(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', newline='') as file:
            reader = csv.reader(file)
            return list(reader)
    else:
        return []

def save_data_to_csv(filepath, data):
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def main():
    # Manually set these variables.
    language_name = 'python'
    libraries = [
        'ninjax',
        'seaborn',
        # 'virtualenv',
        # 'cachetools',
        # 'wrapt',
        # 'jsonschema',
        # 'pluggy',
        # 'sqlalchemy',
        # 'aiohttp',
        # 'tomli',
        # 'pyparsing',
        # 'flask',
        # 'pyarrow',
        # 'yarl',
        # 'multidict',
        # 'pytest',
    ]
    
    max_chars_per_page = 10000
    max_pages = 2

    filepath = 'labeled_data.csv'
    data = load_existing_data(filepath)

    library_data = []

    def scrape_library_info_thread(library_name, language_name, max_chars_per_page, max_pages):
        content = get_doc_data(library_name, language_name)['content']
        print(f"Scraped {len(content)} pages for {library_name}")
        content = [c[:max_chars_per_page] for c in content[:max_pages]]
        library_data.extend(content)

    scrape_threads = []
    for library in libraries:
        scrape_library_info_thread(library, language_name, max_chars_per_page, max_pages)
    #     thread = Thread(target=scrape_library_info_thread, args=(library, language_name, max_chars_per_page, max_pages))
    #     thread.start()
    #     scrape_threads.append(thread)
    
    # for thread in scrape_threads:
    #     thread.join()

    page_idx = 0
    while any(thread.is_alive() for thread in scrape_threads) or page_idx < len(library_data):
        if len(library_data) == 0:
            time.sleep(1)
            continue

        page = library_data.pop(0)
        print('\n\n' + '-' * 50)
        if len(page) < 2000:
            print("Document content:")
            print(page)
        else:
            print("Beginning of the document:")
            print(page[:1000] + "...")
            print("End of the document:")
            print("..." + page[-1000:])
            
        label = None
        while label not in ['1', '2', '3']:
            label = input("Label (1 - Documentation, 2 - Code, 3 - Irrelevant): ")
        data.append([library, language_name, page, label])

        if len(data) % 10 == 0:
            save_data_to_csv(filepath, data)

        page_idx += 1
        scrape_threads = [thread for thread in scrape_threads if thread.is_alive()]

    save_data_to_csv(filepath, data)

if __name__ == '__main__':
    main()
