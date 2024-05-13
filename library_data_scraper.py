import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings

    
class PyPILibrarySpider(scrapy.Spider):
    name = 'pypi_libraries'
    allowed_domains = ['pypi.org', 'github.com']
    start_urls = ['https://pypi.org/search/?c=Development+Status+%3A%3A+5+-+Production%2FStable&c=Natural+Language+%3A%3A+English&c=Programming+Language+%3A%3A+Python+%3A%3A+3&o=-created&q=&page=1']

    def parse(self, response):
        # Extract library elements
        for library_element in response.css('a.package-snippet'):
            library_name = library_element.css('span.package-snippet__name::text').get().strip()
            link = response.urljoin(library_element.attrib['href'])
            yield scrapy.Request(link, callback=self.parse_library_page, meta={'library_name': library_name, 'link': link})

        # Follow pagination link
        next_page = response.css('a.button.button-group__button::attr(href)').getall()[-1]
        if next_page:
            yield response.follow(next_page, self.parse)

    def parse_library_page(self, response):
        library_name = response.meta['library_name']
        link = response.meta['link']
        github_link_selector = 'a.vertical-tabs__tab[href*="github.com"][rel="nofollow"]'
        github_link_element = response.css(github_link_selector).getall()
        print('Github link element:', github_link_element)
        github_link = None
        for element in github_link_element:
            if 'Repository' in element:
                print('selected repository: ', element)
                github_link = element.attrib['href']
                print('selected github link: ', github_link)
                break
        if github_link:
            yield scrapy.Request(github_link, callback=self.parse_github_page, meta={'library_name': library_name, 'link': link})
        else:
            yield {
                'name': library_name,
                'link': link,
                'stars': 'N/A',
            }

    def parse_github_page(self, response):
        stars = response.css('a[href$="/stargazers"] .text-bold::text').get().strip().replace('k', '000')
        yield {
            'name': response.meta['library_name'],
            'link': response.meta['link'],
            'stars': stars if stars else 'N/A',
        }


if __name__ == '__main__':
    settings = Settings({
    'BOT_NAME': 'pypi_scrapy',
    'DOWNLOAD_DELAY': 1,
    # name of the file
    'FEEDS': {
        'pypi_libraries.json': {
            'format': 'json',
            'encoding': 'utf8',
        },
    },
    })

    process = CrawlerProcess(settings=settings)
    process.crawl(PyPILibrarySpider)
    process.start()

