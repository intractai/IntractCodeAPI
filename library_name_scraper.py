import json
import re

import requests
from bs4 import BeautifulSoup


def _reduce_github_url(url: str) -> str:
    match = re.match(r'(https://github\.com/[^/]+/[^/]+)', url)
    if match:
        return match.group(1)
    else:
        return url  # Return the original URL if it doesn't match the pattern

def _is_github_repo_link(url: str) -> bool:
    return re.match(r'https://github\.com/[^/]+/[^/]+', url) is not None

def get_github_link(package_page: str) -> str:
    package_site = requests.get(package_page)
    package_soup = BeautifulSoup(package_site.text, 'html.parser')
    github_link_elements = package_soup.select('li a.vertical-tabs__tab--with-icon[href*="github.com"][rel="nofollow"]')
    github_link = None
    for element in github_link_elements:
        if element.select_one('i[class*="fa-github"]'):
            github_link = element['href']
    if github_link is None:
        for element in github_link_elements:
            if element.select_one('i[class*="fa-home"]') or \
                element.select_one('i[class*="fa-book"]') and \
                    _is_github_repo_link(element['href']):
                github_link = element['href']
                break
    return github_link


def get_github_stars(github_link: str) -> str:
    github_response = requests.get(github_link)
    github_soup = BeautifulSoup(github_response.text, 'html.parser')
    stars_element = github_soup.select_one('a[href*="/stargazers"] .text-bold')
    if stars_element:
        stars = stars_element.text.strip().replace('k', '000') # Assuming 'k' stands for thousands
    else:
        stars = 'N/A'
    return stars


def scrape_libraries():
    start_url = 'https://pypi.org/search/?c=Development+Status+%3A%3A+5+-+Production%2FStable&c=Natural+Language+%3A%3A+English&c=Programming+Language+%3A%3A+Python+%3A%3A+3&o=-created&q=&page=1'
    libraries = []

    while True:
        response = requests.get(start_url)
        print('Searching page:', start_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        library_elements = soup.select('a.package-snippet')
        try:
            for library_element in library_elements:
                library_name = library_element.select_one('span.package-snippet__name').text
                link = library_element.attrs['href']
                package_page = 'https://pypi.org' + link
                print('Package page:', package_page)
                github_link = get_github_link(package_page)
                stars = 'N/A'
                if github_link:
                    print('Github link:', github_link)
                    github_link = _reduce_github_url(github_link)
                    stars = get_github_stars(github_link)
                    print('Stars:', stars)
                else:
                    print('No github link found')
                libraries.append({'name': library_name, 'link': link, 'stars': stars})
        except Exception as e:
            print('Error:', e)

        next_page_element = soup.select_one('a.button.button-group__button:contains("Next")')
        if next_page_element and 'href' in next_page_element.attrs:
            try: 
                next_page_url = 'https://pypi.org' + next_page_element.attrs['href']
                start_url = next_page_url
            except Exception as e:
                print('Error:', e)
                break
        else:
            break

    with open('test.json', 'w') as f:
        json.dump(libraries, f, indent=4)

if __name__ == '__main__':
    scrape_libraries()
