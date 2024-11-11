import hashlib
import logging
import pickle
from functools import cache

import argostranslate.package
import argostranslate.translate
from google.cloud import translate_v2

import config


class ArgosTranslateProvider:
    def __init__(self, source_lang: str, target_lang: str) -> None:
        self.source_lang = source_lang
        self.target_lang = target_lang

        installed_languages = {x.code for x in argostranslate.translate.get_installed_languages()}
        if source_lang in installed_languages and target_lang in installed_languages:
            logging.debug('Languages for the translator are already installed.')
            return

        logging.debug('Languages for the translator are not found. Let\'s install them.')
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        package_to_install = next(filter(lambda x: x.from_code == source_lang and x.to_code == target_lang, available))
        argostranslate.package.install_from_path(package_to_install.download())

    @cache
    def translate(self, text: str) -> str:
        '''Generate new third-party translation.'''
        return argostranslate.translate.translate(text, self.source_lang, self.target_lang)


class GCTranslationProvider:
    def __init__(self, source_lang: str, target_lang: str) -> None:
        self.client = translate_v2.Client()
        self.source_lang = source_lang
        self.target_lang = target_lang

    @cache
    def translate(self, text: str) -> str:
        response = self.client.translate(text, source_language=self.source_lang, target_language=self.target_lang)
        return response['translatedText']


class Translator:
    def __init__(self, container: 'DependencyContainer') -> None:
        self.aligner = container.aligner
        self.text_processor = container.text_processor
        match config.translation_provider:
            case 'GoogleCloud':
                provider = GCTranslationProvider
            case 'Argos':
                provider = ArgosTranslateProvider
        self.provider = provider(config.source_lang, config.target_lang)

    def translate(self, text: str) -> str:
        return self.provider.translate(text)

    @staticmethod
    def _get_stable_hash(text: str) -> str:
        '''Helper for getting stable hash for the same sentences between sessions.'''
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def process_sentences(self, original_sentences: list[str], use_translation_file: int) -> list[str]:
        '''Process all sentences according to current config rules.'''
        if use_translation_file == 1:
            result_src, result_trg = self.get_literary_translation_from_file(original_sentences)
            self.translated = {self._get_stable_hash(src): trg for src, trg in zip(result_src, result_trg)}
            return result_src
        if use_translation_file == 2:
            with open('translation.pkl', 'rb') as file:
                self.translated = pickle.load(file)
            return original_sentences
        self.translated = {self._get_stable_hash(src): self.translate(src.rstrip()) for src in original_sentences}
        return original_sentences

    def save_translation_to_file(self) -> None:
        '''Save translation in order to reduce the number of requests to external translators.'''
        with open('translation.pkl', 'wb') as file:
            pickle.dump(self.translated, file)

    def get_translated_sentence(self, sentence: str) -> str:
        return self.translated[self._get_stable_hash(sentence)]

    def get_literary_translation_from_file(self, original_sentences: list[str]) -> tuple[list[str], list[str]]:
        '''Use existing translation from file as parallel text.'''
        with open('translation.txt', 'r') as file:
            text = file.read()
        translated_sentences = self.text_processor.get_sentences(text)
        result = []
        current_src, current_trg = '', ''
        i, j = 0, 0
        s, t = len(original_sentences) - 1, len(translated_sentences) - 1
        while i < s and j < t:
            next_src_score = self.get_sentence_similarity_score(original_sentences[i + 1], translated_sentences[j])
            next_trg_score = self.get_sentence_similarity_score(original_sentences[i], translated_sentences[j + 1])
            next_both_score = self.get_sentence_similarity_score(original_sentences[i + 1], translated_sentences[j + 1])
            if next_src_score > next_both_score:
                current_src += f' {original_sentences[i]}'
                i += 1
            elif next_trg_score > next_both_score:
                current_trg += f' {translated_sentences[j]}'
                j += 1
            else:
                result.append((current_src, current_trg))
                current_src = original_sentences[i]
                current_trg = translated_sentences[j]
                i += 1
                j += 1
        if current_src and current_trg:
            result.append((current_src, current_trg))
        if i == s and j == t:
            result.append((original_sentences[i], translated_sentences[j]))
        return zip(*result)

    def get_sentence_similarity_score(self, sentence_src: str, sentence_trg: str) -> float:
        '''Simple similarity scoring based on percent of inner-aligned words.'''
        alignment = self.aligner.get_word_aligns(sentence_src, sentence_trg)
        aligned_words_count = len(alignment['inter'])
        total_words = max(len(sentence_src.split()), len(sentence_trg.split()))
        return aligned_words_count / total_words
