import logging
import regex as re

from simalign import SentenceAligner
from spacy.tokens import Token

from _structures import Entity, Lemma, LemmaTrie
from alignment import TokenAligner
from config import Config
from synthesis import SpeechSynthesizer
from text_processing import TextProcessing
from translation import Translator

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class DependencyContainer:
    def __init__(self):
        self.config = Config()
        self.text_processor = TextProcessing(self.config)
        self.aligner = SentenceAligner(model='xlm-roberta-base', token_type='bpe', matching_methods='a')
        self.translator = Translator(self.config.source_lang, self.config.target_lang)
        self.synthesizer = SpeechSynthesizer(
            self.config.source_lang, self.config.target_lang, self.config.lang_delimeter
        )
        self.lemma_trie = LemmaTrie()
        self.lemmas = {}

    def get(self, dependency_name: str) -> object:
        return getattr(self, dependency_name)


class Main:
    def __init__(self) -> None:
        self.container = DependencyContainer()
        self.sentence_counter = 0
        self.entity_counter = 0
        self.output_text: list[str] = []

    def process(self, text: str) -> None:
        sentences = self.container.text_processor.get_sentences(text)
        for sentence_text in sentences:
            sentence_obj = Sentence(sentence_text, self.entity_counter, self.container)
            sentence_obj.process_tokens()
            self.entity_counter = sentence_obj.current_pos
            self.sentence_counter += 1
            self.output_text.append(sentence_obj.get_result_line())

        self.container.synthesizer.save_tts(''.join(self.output_text))


class Sentence:
    def __init__(self, sentence: str, current_pos: int, container: DependencyContainer) -> None:
        self.container = container
        self.current_pos = current_pos
        self.sentence = sentence
        self.translated_sentence = container.translator.translate(sentence)

        self.tokens_src = container.text_processor.get_merged_tokens(
            self.sentence, lang=container.config.source_lang
        )
        self.tokens_trg = container.text_processor.get_merged_tokens(
            self.translated_sentence, lang=container.config.target_lang
        )
        logging.info(f'{self.tokens_src=}, {self.tokens_trg=}')
        container.text_processor.add_tokens_embeddings(self.sentence, self.tokens_src)
        container.text_processor.add_tokens_embeddings(self.translated_sentence, self.tokens_trg)

        self.aligner = TokenAligner(container, self)

        self.result: list[tuple[str, str]] = []
        self.possible_result: list[tuple[str, str]] = []
        self.skip: set[Token] = set()

    def process_tokens(self) -> None:
        for idx_src, token_src in enumerate(self.tokens_src):
            # TODO POS aligning
            self.current_pos += 1
            logging.debug(f'Processing token: {token_src}')
            if token_src in self.skip:  # skip token if it has already been processed
                logging.debug('Skipping previously processed token')
            elif not re.match(self.container.config.punct_pattern, token_src.text):  # skip punct w/o counting
                self.current_pos -= 1
                logging.debug('Skipping punctuation')
            elif token_src.ent_type_ in self.container.config.untranslatable_entities:
                logging.debug(f'Skipping untranslatable entity: {token_src.ent_type_}')
            elif self._is_start_of_named_entity(idx_src):  # treat named entity chain and add tokens to 'skip'
                self.search_named_entity(idx_src)
            elif self.trie_search(idx_src):  # check multiword origin chain
                logging.debug('Found multiword chain')
            elif token_src.text.lower() in self.container.config.stop_words:  # only after multiword check
                logging.debug('Skipping stopword')
            else:
                self.aligner.process_alignment(idx_src)
        logging.info(f'Result: {self.result}, Possible Result: {self.possible_result}')

    def _is_start_of_named_entity(self, idx_src: int) -> bool:
        return (
            self.tokens_src[idx_src].ent_iob_ == 'B'
            and len(self.tokens_src) > idx_src + 1
            and self.tokens_src[idx_src + 1].ent_iob_ == 'I'
        )

    def search_named_entity(self, idx_src: int) -> None:
        token_seq = [self.tokens_src[idx_src]]
        for forw_token in self.tokens_src[idx_src + 1 :]:
            if forw_token.ent_iob_ != 'I':
                break
            token_seq.append(forw_token)

        translation = self.container.translator.translate(' '.join(token.text for token in token_seq))
        ent_obj = Entity(None, self.container.config.entity_intervals, translation)
        entity = self.container.lemma_trie.add([t.lemma_ for t in token_seq], ent_obj)
        if entity.check_repeat(self.current_pos + idx_src):
            entity.update(self.current_pos + idx_src)
            self.result.append((' '.join(token.text for token in token_seq), translation))
        self.skip |= set(token_seq)
        logging.debug(f'Named entity found and processed: {token_seq}')

    def trie_search(self, idx_src: int) -> bool:
        ent, depth, punct_tail = self.container.lemma_trie.search(self.tokens_src, idx_src)
        if ent and depth - idx_src - punct_tail > 1:
            if ent.check_repeat(self.current_pos + idx_src):
                ent.update(self.current_pos + idx_src)
                result_text = ' '.join(token.text for token in self.tokens_src[idx_src : depth - punct_tail])
                self.result.append((result_text, ent.translation))
            self.skip |= set(self.tokens_src[idx_src : depth - punct_tail])
            logging.debug(
                'Known multiword chain found: {}'.format(
                    ' '.join(self.tokens_src[x].text for x in range(idx_src, depth - punct_tail))
                )
            )
            return True
        logging.debug('No multiword chain found')
        return False

    def get_result_line(self) -> str:
        result = self.sentence
        delim = self.container.config.lang_delimeter
        if self.result:
            result += ' [{}] '.format(', '.join(f'{src}–{delim}{trg}{delim}' for src, trg in self.result))
            # TODO add possible result?
            n = len(self.result)
            if n > 6 or 2 < n > len(self.tokens_src) // 3:
                result += f'{delim}{self.translated_sentence}{delim} '

        return result


if __name__ == '__main__':
    app = Main()
    with open('text.txt', 'r') as file:
        text = file.read()
    app.process(text)
