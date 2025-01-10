import logging
import pickle
from hashlib import sha256
from html import unescape

import argostranslate.package
import argostranslate.translate
import config
import dependencies as container
from google.cloud import translate_v2
from sentence_transformers import SentenceTransformer


class ArgosTranslateProvider:
    '''Local translator without any requirements.'''

    def __init__(self) -> None:
        installed_languages = {x.code for x in argostranslate.translate.get_installed_languages()}
        if config.source_lang in installed_languages and config.target_lang in installed_languages:
            logging.debug('Languages for the translator are already installed.')
            return

        logging.debug('Languages for the translator are not found. Let’s install them.')
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        package_to_install = next(
            filter(lambda x: x.from_code == config.source_lang and x.to_code == config.target_lang, available)
        )
        argostranslate.package.install_from_path(package_to_install.download())

    def translate(self, text: str) -> str:
        '''Generate new third-party translation.'''
        return argostranslate.translate.translate(text, config.source_lang, config.target_lang)


class GCTranslateProvider:
    '''Google Cloud translator. Requires GOOGLE_APPLICATION_CREDENTIALS env.var with path to your credentials file.'''

    def __init__(self) -> None:
        self.client = translate_v2.Client()

    def translate(self, text: str) -> str:
        '''Generate new third-party translation.'''
        response = self.client.translate(text, source_language=config.source_lang, target_language=config.target_lang)
        return response['translatedText']


class Translator:
    def __init__(self) -> None:
        self.translated: dict[str, str] = {}
        match config.translation_provider:
            case 'GoogleCloud':
                self.provider = GCTranslateProvider()
            case 'Argos':
                self.provider = ArgosTranslateProvider()  # type: ignore
            case value:
                raise ValueError(f'Unknown translation_provider value ({value}).')

        if config.use_translation_file == 1:
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def translate(self, text: str) -> str:
        '''Delegate the translation to the chosen provider.'''
        if (text_hash := self._get_stable_hash(text)) not in self.translated:
            self.translated[text_hash] = unescape(self.provider.translate(text))
        return self.translated[text_hash]

    @staticmethod
    def _get_stable_hash(text: str) -> str:
        '''Get stable hash for the same sentences between sessions.'''
        return sha256(text.encode('utf-8')).hexdigest()

    def process_sentences(self, sentences: list[str]) -> list[str]:
        '''Process all sentences according to current config rules. Return possibly changed list of sentences.'''
        match config.use_translation_file:
            case 0:
                self.translated = {self._get_stable_hash(src): self.translate(src.rstrip()) for src in sentences}
            case 1:
                src_sentences, trg_sentences = self.get_literary_translation_from_file(sentences)
                self.translated = {self._get_stable_hash(src): trg for src, trg in zip(src_sentences, trg_sentences)}
                return src_sentences
            case 2:
                with open(config.root_dir / 'temp' / 'translation.pkl', 'rb') as file:
                    self.translated = pickle.load(file)  # nosec
        return sentences

    def get_translated_sentence(self, sentence: str) -> str:
        '''Return existing translation by hash of the original sentence.'''
        return self.translated[self._get_stable_hash(sentence)]

    def save_translation_to_file(self) -> None:
        '''Save the translation to reduce the number of requests to external translators.'''
        with open(config.root_dir / 'temp' / 'translation.pkl', 'wb') as file:
            pickle.dump(self.translated, file)

    def get_literary_translation_from_file(self, original_sentences: list[str]) -> tuple[list[str], list[str]]:
        '''Use existing translation from file as parallel text. Pretty unreliable right now ¯|_(ツ)_/¯.'''
        # TODO improve
        with open(config.root_dir / 'input_translation.txt', 'r', encoding='utf-8') as file:
            text = file.read()
        translated_sentences = container.text_preprocessor.get_sentences(text)
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

        result: list[list[str]] = []
        prev_src, prev_trg = -1, -1
        for src, trg in result_idxs:
            if src == prev_src:
                result[-1][1] += f' {translated_sentences[trg]}'
            elif trg == prev_trg:
                result[-1][0] += f' {original_sentences[src]}'
            else:
                result.append([original_sentences[src], translated_sentences[trg]])
            prev_src, prev_trg = src, trg

        return zip(*result)  # type: ignore
