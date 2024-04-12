from abc import ABC, abstractmethod
import os
import logging

import jinja2
import yaml
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_fixed

from src.crawler.docs_scraper import get_docs_overview
from src import config_handler


logger = logging.getLogger(__name__)


class AutoDataGenerator(ABC):

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    def _generate_helper(self, user_content: str, system_content: str, temperature: float):
        response = completion(
            model=self._model_name, 
            messages=[
                {"content": system_content, "role": "system"},
                {"content": user_content, "role": "user"},
                ],
            temperature=temperature
        )
        return response.choices[0].message.content


class LibraryProblemGenerator(AutoDataGenerator):
    
    def __init__(self, model_name: str, lang: str, library: str, max_chars: int, 
                 feature_num: int, problem_num_per_bullet_point: int):
        self._model_name = model_name
        self._lang = lang
        self._library = library
        self._max_chars = max_chars #TODO: not used for now.
        self._feature_num = feature_num
        self._problem_num_per_bullet_point = problem_num_per_bullet_point

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def _generate_doc_description(self, cfg: dict, doc_info: str):
        
        environment = jinja2.Environment()
        template = environment.from_string(cfg['system'])
        system_content = template.render(library_docs=doc_info, language_name=self._lang, 
                                        library=self._library, feature_num=self._feature_num)
        template = environment.from_string(cfg['user'])
        user_content = template.render(library_docs=doc_info, language_name=self._lang, 
                                        library=self._library, feature_num=self._feature_num)

        problem_description = self._generate_helper(
            user_content=user_content, 
            system_content=system_content,
            temperature=cfg['temperature']
        )
        return problem_description

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def _generate_problems_description(self, cfg: dict, doc_desc: str):    
        environment = jinja2.Environment()
        template = environment.from_string(cfg['system'])
        system_content = template.render(library_desc=doc_desc, language_name=self._lang, 
                                        library=self._library, feature_num=self._feature_num,
                                        problem_num=self._problem_num_per_bullet_point)
        template = environment.from_string(cfg['user'])
        user_content = template.render(library_desc=doc_desc, language_name=self._lang, 
                                    library=self._library, feature_num=self._feature_num,
                                    problem_num=self._problem_num_per_bullet_point)
        
        problems = self._generate_helper(
            user_content=user_content, 
            system_content=system_content,
            temperature=cfg['temperature']
        )

        # post processing to remove the explanation
        problems = yaml.safe_load(problems[8:-3])
        problems = problems['problems']
        return problems

    def generate(self):
        cfg = config_handler.get_config()
        doc_info = get_docs_overview(self._library, self._lang)
        logger.info(f"FINISHED: Extracting {self._library} documentation information.")
        doc_desc = self._generate_doc_description(cfg['describe_library_doc'], doc_info)
        logger.info(f"FINISHED: Generating {self._library} description.")
        problems = self._generate_problems_description(cfg['generate_library_problems'], doc_desc)
        logger.info(f"FINISHED: Generating {self._library} problems.")
        return problems
