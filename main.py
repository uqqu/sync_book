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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


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
        '''Load LemmaDict and LemmaTrie from a previously saved file.'''
        with open(f'{self.config.input_storage_filename}.pkl', 'rb') as file:
            self.lemma_dict, self.lemma_trie = pickle.load(file)


class Main:
    def __init__(self) -> None:
        self.container = DependencyContainer()
        self.sentence_counter = 0
        self.entity_counter = 0
        # int: 0 – source sentence, 1 – target sentence, 2 – vocabulary, 3 – whitespaces
        self.output_text: list[tuple[int, str | list[tuple[str, str]]]] = []
        self.output_ssml: list[str] = ['<speak>']
        if not Token.has_extension('embedding'):
            Token.set_extension('embedding', default=None)

        if self.container.config.speech_synth and self.container.config.speech_config.mfa_use:
            self.output_audio: 'AudioSegment' = self.container.synthesizer.silent(0)
            if not Token.has_extension('audio'):
                Token.set_extension('audio', default=self.container.synthesizer.silent(200))

    def process(self, text: str) -> None:
        '''Main app cycle.'''
        sentences = self.container.text_processor.get_sentences(text)
        # it may be changed after the sentence alignment with the literary translation
        sentences = self.container.translator.process_sentences(sentences, self.container.config.use_translation_file)
        for sentence_text in sentences:
            sentence_obj = Sentence(sentence_text, self.entity_counter, self.container)
            sentence_obj.process_tokens()
            self.entity_counter = sentence_obj.entity_counter
            self.sentence_counter += 1
            self.output_ssml.append(sentence_obj.get_rendered_ssml())
            self.output_text.extend(sentence_obj.get_results())
            if self.container.config.speech_synth and self.container.config.speech_config.mfa_use:
                self.output_audio += sentence_obj.get_result_mfa_audio()

        self.output_ssml.append('</speak>')

        if self.container.config.speech_synth:
            if not self.container.config.speech_config.mfa_use:
                if self.container.config.speech_config.ssml:
                    self.output_audio = self.container.synthesizer.synthesize(''.join(self.output_ssml), '', 1)
                else:
                    self.output_audio = self.container.synthesizer.synthesize_by_parts(
                        self.output_text, self.container.config.speech_config.sentence_speed
                    )
            self.container.synthesizer.save_audio(self.output_audio, 'multilingual_output')
        self.container.save_structures()


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
