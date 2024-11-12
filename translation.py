import hashlib
import logging
import pickle
from functools import cache

import argostranslate.package
import argostranslate.translate
from google.cloud import translate_v2
from sentence_transformers import SentenceTransformer

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
        if config.use_translation_file == 1:
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

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
        sentences = original_sentences + translated_sentences
        embeddings = self.model.encode(sentences)
        similarities = self.model.similarity(embeddings, embeddings)

        result_idxs = []
        nh, nf = len(original_sentences), len(sentences)
        src_idx, trg_idx = 0, nh
        while src_idx < nh and trg_idx < nf:
            next_src_score = similarities[src_idx + 1][trg_idx] if src_idx + 1 < nh else 0
            next_trg_score = similarities[src_idx][trg_idx + 1] if trg_idx + 1 < nf else 0
            next_both_score = similarities[src_idx + 1][trg_idx + 1] if src_idx + 1 < nh and trg_idx + 1 < nf else 0
            result_idxs.append((src_idx, trg_idx - nh))
            if next_src_score > next_both_score:
                src_idx += 1
            elif next_trg_score > next_both_score:
                trg_idx += 1
            else:
                src_idx += 1
                trg_idx += 1

        result = []
        prev_src, prev_trg = -1, -1
        for src, trg in result_idxs:
            if src == prev_src:
                result[-1][1] += f' {translated_sentences[trg]}'
            elif trg == prev_trg:
                result[-1][0] += f' {original_sentences[src]}'
            else:
                result.append([original_sentences[src], translated_sentences[trg]])
            prev_src, prev_trg = src, trg

        return zip(*result)
