import logging
from functools import cache

import argostranslate.package
import argostranslate.translate


class Translator:
    def __init__(self, container: 'DependencyContainer') -> None:
        self.aligner = container.aligner
        self.source_lang = container.config.source_lang
        self.target_lang = container.config.target_lang
        self.text_processor = container.text_processor

        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        package_to_install = next(
            filter(lambda x: x.from_code == self.source_lang and x.to_code == self.target_lang, available)
        )
        argostranslate.package.install_from_path(package_to_install.download())

    @cache
    def translate(self, text: str) -> str:
        '''Generate new third-party translation.'''
        return argostranslate.translate.translate(text, self.source_lang, self.target_lang)

    def process_sentences(self, original_sentences: list[str], use_translation_file: bool) -> list[str]:
        '''Process all sentences according to current config rules.'''
        if use_translation_file:
            result_src, result_trg = self.get_from_file(original_sentences)
            self.translated = {hash(src): trg for src, trg in zip(result_src, result_trg)}
            return result_src
        self.translated = {hash(src): self.translate(src) for src in original_sentences}
        return original_sentences

    def get_translated_sentence(self, sentence: str) -> str:
        return self.translated[hash(sentence)]

    def get_sentence_similarity_score(self, sentence_src: str, sentence_trg: str) -> float:
        '''Simple similarity scoring based on percent of inner-aligned words.'''
        alignment = self.aligner.get_word_aligns(sentence_src, sentence_trg)
        aligned_words_count = len(alignment['inter'])
        total_words = max(len(sentence_src.split()), len(sentence_trg.split()))
        return aligned_words_count / total_words

    def get_from_file(self, original_sentences: list[str]) -> tuple[list[str], list[str]]:
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
