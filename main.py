import logging
import os
import pickle

from simalign import SentenceAligner
from spacy.tokens import Token

from _structures import LemmaDict, LemmaTrie
from config import Config
from sentence_processing import Sentence
from synthesis import SpeechSynthesizer
from text_processing import TextProcessing
from translation import Translator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DependencyContainer:
    def __init__(self):
        self.config = Config()
        self.aligner = SentenceAligner(model='xlm-roberta-base', token_type='bpe', matching_methods='a')
        self.synthesizer = SpeechSynthesizer(self.config)
        self.text_processor = TextProcessing(self.config)
        self.translator = Translator(self)

        try:
            self.load_structures()
        except FileNotFoundError:
            self.lemma_dict = LemmaDict()
            self.lemma_trie = LemmaTrie()

    def get(self, dependency_name: str) -> object:
        return getattr(self, dependency_name)

    def save_structures(self) -> None:
        '''Save LemmaDict and LemmaTrie structures to file for future reuse. Keep previous version if exists.'''
        if not self.config.save_results:
            return

        filename = f'{self.config.output_storage_filename}.pkl'
        if os.path.exists(filename):
            old_filename = f'{filename}.old'
            if os.path.exists(old_filename):
                os.remove(old_filename)
            os.rename(filename, old_filename)
        with open(filename, 'wb') as file:
            pickle.dump((self.lemma_dict, self.lemma_trie), file)

    def load_structures(self) -> None:
        '''Load LemmaDict and LemmaTrie from a file.'''
        with open(f'{self.config.input_storage_filename}.pkl', 'rb') as file:
            self.lemma_dict, self.lemma_trie = pickle.load(file)


class Main:
    def __init__(self) -> None:
        self.container = DependencyContainer()
        self.sentence_counter = 0
        self.entity_counter = 0
        self.output_text: list[str] = []
        self.output_audio: 'AudioSegment' = self.container.synthesizer.silent(0)
        if not Token.has_extension('embedding'):
            Token.set_extension('embedding', default=None)
        if not Token.has_extension('audio'):
            Token.set_extension('audio', default=self.container.synthesizer.silent(200))

    def process(self, text: str) -> None:
        '''Main app cycle.'''
        sentences = self.container.text_processor.get_sentences(text)
        # it may be changed after the sentence alignment with the literary translation
        sentences = self.container.translator.process_sentences(sentences, self.container.config.use_translation_file)
        for sentence_text in sentences:
            sentence_obj = Sentence(sentence_text, self.entity_counter, self.container)
            sentence_obj.process_audio()
            sentence_obj.process_tokens()
            self.entity_counter = sentence_obj.current_pos
            self.sentence_counter += 1
            self.output_text.append(sentence_obj.get_result_text())
            if self.container.config.speech_synth:
                self.output_audio = self.container.synthesizer.add(self.output_audio, sentence_obj.get_result_audio())

        if self.container.config.speech_synth:
            self.container.synthesizer.save_audio(self.output_audio, 'multilingual_output')
        self.container.save_structures()


if __name__ == '__main__':
    app = Main()
    with open('text.txt', 'r') as file:
        text = file.read()
    app.process(text)
    print(''.join(app.output_text))
