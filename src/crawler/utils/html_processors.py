import trafilatura

def extract_main_text_from_html(txt: str):
    extracted_text = trafilatura.extract(txt, output_format='txt',
                                         include_comments=True, include_tables=True)
    return extracted_text if extracted_text else ""