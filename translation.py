import logging
from functools import cache

import argostranslate.package
import argostranslate.translate


class Translator:
    def __init__(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        package_to_install = next(
            filter(lambda x: x.from_code == self.source_lang and x.to_code == self.target_lang, available)
        )
        argostranslate.package.install_from_path(package_to_install.download())

    @cache
    def translate(self, text):
        return argostranslate.translate.translate(text, self.source_lang, self.target_lang)
