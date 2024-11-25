import logging
import os
import pickle

from simalign import SentenceAligner
from spacy.tokens import Token

import config
from _structures import LemmaDict, LemmaTrie
from sentence_processing import Sentence
from synthesis import SpeechSynthesizer
from text_processing import TextProcessing
from translation import Translator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DependencyContainer:
    def __init__(self) -> None:
        method = {'inter': 'a', 'mwmf': 'm', 'itermax': 'i', 'fwd': 'f', 'rev': 'r'}[config.alignment_matching_method]
        self.aligner = SentenceAligner(model='xlm-roberta-base', token_type='bpe', matching_methods=method)
        self.synthesizer = SpeechSynthesizer()
        self.text_processor = TextProcessing()
        self.translator = Translator(self)

        try:
            self.load_structures()
        except FileNotFoundError:
            self.lemma_dict = LemmaDict()
            self.lemma_trie = LemmaTrie()
            self.entity_counter = 0
            self.sentence_counter = 0

    def save_structures(self) -> None:
        '''Save structures and progress to file for future reuse. Keep previous version if exists.'''
        if not config.save_results:
            return

        filename = f'{config.output_storage_filename}.pkl'
        if os.path.exists(filename):
            old_filename = f'{filename}.old'
            if os.path.exists(old_filename):
                os.remove(old_filename)
            os.rename(filename, old_filename)
        with open(filename, 'wb') as file:
            pickle.dump((self.lemma_dict, self.lemma_trie, self.entity_counter, self.sentence_counter), file)

    def load_structures(self) -> None:
        '''Load structures and progress from a previously saved file.'''
        with open(f'{config.input_storage_filename}.pkl', 'rb') as file:
            self.lemma_dict, self.lemma_trie, self.entity_counter, self.sentence_counter = pickle.load(file)

    def print_structures(self):
        '''Representation of the structures for debug and control.'''

        def _print_trie(elem, spaces=0):
            for word, child in elem.items():
                if isinstance(child, LemmaTrie):
                    print(f'{" " * spaces}{word}')
                    _print_trie(child.children, spaces + len(word) + 1)
                else:
                    print(f'{" " * spaces}{child.translation=} {child.level=} {child.last_pos=}')

        # lemma_trie
        _print_trie(self.lemma_trie.children)
        # lemma_dict
        for lemma, child in self.lemma_dict.children.items():
            print(lemma)
            spaces = len(lemma)
            for form, entity in child.children.items():
                print(' ' * spaces, form, entity.translation, entity.level, entity.last_pos)


class Main:
    def __init__(self) -> None:
        self.container = DependencyContainer()
        # int: 0 – source sentence, 1 – target sentence, 2 – vocabulary, 3 – whitespaces
        self.output_text: list[tuple[int, str | list[tuple[str, str]]]] = []
        self.output_ssml: list[str] = ['<speak>']
        if not Token.has_extension('embedding'):
            Token.set_extension('embedding', default=None)

        if config.speech_synth and config.use_mfa:
            self.output_audio: 'AudioSegment' = self.container.synthesizer.silent(0)
            if not Token.has_extension('audio'):
                Token.set_extension('audio', default=self.container.synthesizer.silent(200))

    def process(self, text: str) -> None:
        '''Main app cycle.'''
        sentences = self.container.text_processor.get_sentences(text)
        # it may be changed after the sentence alignment with the literary translation
        sentences = self.container.translator.process_sentences(sentences, config.use_translation_file)
        if config.save_translation_to_file:
            self.container.translator.save_translation_to_file()
        for sentence_text in sentences:
            sentence_obj = Sentence(sentence_text, self.container.entity_counter, self.container)
            sentence_obj.process_tokens()
            self.container.entity_counter = sentence_obj.entity_counter
            self.container.sentence_counter += 1
            self.output_ssml.append(sentence_obj.get_rendered_ssml())
            self.output_text.extend(sentence_obj.get_results())
            if config.speech_synth and config.use_mfa:
                self.output_audio += sentence_obj.get_result_mfa_audio() + self.container.synthesizer.silent(100)

        self.output_ssml.append('</speak>')

        if config.speech_synth:
            if not config.use_mfa:
                if config.use_ssml:
                    self.output_audio = self.container.synthesizer.synthesize(''.join(self.output_ssml), '', 1)
                else:
                    self.output_audio = self.container.synthesizer.synthesize_by_parts(
                        self.output_text, config.sentence_pronunciation_speed
                    )
            self.container.synthesizer.save_audio(self.output_audio, 'multilingual_output')
        self.container.save_structures()
        # self.container.print_structures()


if __name__ == '__main__':
    app = Main()
    with open('text.txt', 'r') as file:
        text = file.read()
    app.process(text)
    print(
        ' '.join(
            x[1] if isinstance(x[1], str) else f"[{', '.join(f'{a}–{b}' for a, b in x[1])}]" for x in app.output_text
        )
    )
    print(''.join(app.output_ssml))
