from typing import Optional

from googlesearch import search
from litellm import completion


# QUERY_TEMPLATE = "Latest {} library documentation website"
QUERY_TEMPLATE = "{language} {library} documentation"
lib_to_doc = {
    'html2text': 'https://html2text.readthedocs.io/en/latest/',
    'pytorch': 'https://pytorch.org/docs/stable/package.html#tutorials'
}


def find_doc_first_page(library: str, language: Optional[str], model_name: Optional[str] = 'gpt-3.5-turbo'):
    if library in lib_to_doc:
        return lib_to_doc[library]
    query = QUERY_TEMPLATE.format(library=library, language=language)

    if model_name:
        urls = list(search(query, sleep_interval=0, num_results=5, advanced=False, lang='en'))
        # system_content = f"Your job is to find the documentation for the {library} library."
        user_content = [f"{i}: {url}\n" for i, url in enumerate(urls, 1)]
        user_content = ''.join(user_content)
        user_content += \
            f"Select the link that is most likely to have the best {language} " + \
            f"documentation for the {library} library. " + \
            f"Respond with only a number 1-5, or with NA if none of the links are relevant."

        response = completion(
            model = model_name, 
            messages = [
                # {"content": system_content, "role": "system"},
                {"content": user_content, "role": "user"},
            ],
        )
        content = response.choices[0].message.content

        if content.lower().startswith == 'na':
            return None
        elif content[0].isdigit() and 1 <= int(content) <= 5:
            return urls[int(content[0]) - 1]
        elif content in urls:
            return content
        else:
            return None
    else:
        return list(search(query, sleep_interval=0, num_results=1, advanced=False, lang='en'))[0]