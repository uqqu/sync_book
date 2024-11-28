import logging
from itertools import dropwhile

import regex as re
from core._structures import Entity
from core.alignment import TokenAligner
from core.mfa_aligner import MFAligner
from jinja2 import Template

import config


class Sentence:
    def __init__(self, sentence: str, entity_counter: int, container: 'DependencyContainer') -> None:
        self.sentence = sentence
        self.container = container
        self.translated_sentence = self.container.translator.get_translated_sentence(sentence)
        self.entity_counter = entity_counter

        self.tokens_src = container.text_processor.get_merged_tokens(self.sentence, lang=config.source_lang)
        self.tokens_trg = container.text_processor.get_merged_tokens(self.translated_sentence, lang=config.target_lang)
        logging.info(f'{self.tokens_src=}, {self.tokens_trg=}')
        container.text_processor.add_tokens_embeddings(self.sentence, self.tokens_src)
        container.text_processor.add_tokens_embeddings(self.translated_sentence, self.tokens_trg)

        self.aligner = TokenAligner(container, self.tokens_src, self.tokens_trg)
        with open(config._root_dir / 'src' / 'templates' / 'template.min.ssml', 'r', encoding='utf-8') as file:
            self.template = Template(file.read())

        self.mfa_aligner = MFAligner(self)

        self.result: list[tuple[str, str]] = []
        self.possible_result: list[tuple[float, str, str]] = []
        self.skip: set['Token'] = set()

    def process_tokens(self) -> None:
        for idx_src, token_src in enumerate(self.tokens_src):
            self.entity_counter += 1
            logging.debug(f'Processing token: {token_src}')
            if token_src in self.skip:  # skip token if it has already been processed
                logging.debug('Skipping previously processed token')
            elif not re.match(config.word_pattern, token_src.text):  # skip punct w/o counting
                self.entity_counter -= 1
                logging.debug('Skipping punctuation')
            elif token_src.ent_type_ in config.untranslatable_entities or 'Art' in token_src.morph.get('PronType'):
                logging.debug(
                    f'Skipping untranslatable entity or article: {token_src.ent_type_}'
                    f'– {token_src.morph.get("PronType")}'
                )
            elif self.trie_search_and_process(idx_src):  # check multiword origin chain
                logging.debug('Found multiword chain')
            elif self._is_start_of_named_entity(idx_src):  # treat named entity chain and add tokens to 'skip'
                self.add_named_entity_to_trie(idx_src)
            elif token_src.text.lower() in config.stop_words:  # only after multiword check
                logging.debug('Skipping stopword')
            else:
                score, seq_tokens_src, seq_tokens_trg = self.aligner.process_alignment(idx_src)
                seq_tokens_src = list(dropwhile(lambda t: 'Art' in t.morph.get('PronType'), seq_tokens_src))
                seq_tokens_trg = list(dropwhile(lambda t: 'Art' in t.morph.get('PronType'), seq_tokens_trg))
                if not seq_tokens_src or not seq_tokens_trg:
                    continue

                if score < config.min_align_weight:
                    self.possible_result.append((round(score, 2), seq_tokens_src, seq_tokens_trg))
                    logging.debug(f'Rejected after alignment: {score}, {seq_tokens_src}, {seq_tokens_trg}')
                    continue
                logging.debug(f'Approved after alignment: {score}, {seq_tokens_src}, {seq_tokens_trg}')

                if len(seq_tokens_src) == 1:
                    self.treat_dict_entity(seq_tokens_src, seq_tokens_trg)
                else:
                    translation = ' '.join(token.text for token in seq_tokens_trg)
                    entity = self.container.lemma_trie.add(
                        [token.lemma_ for token in seq_tokens_src], Entity(translation)
                    )
                    self.treat_trie_entity(entity, seq_tokens_src, seq_tokens_trg)

        logging.info(f'Result: {self.result}, Possible Result: {self.possible_result}')

    def trie_search_and_process(self, idx_src: int) -> bool:
        '''Look for existing source multiword sequence in LemmaTrie.'''
        entity, depth = self.container.lemma_trie.search(self.tokens_src[idx_src:])
        if depth > 1:
            self.treat_trie_entity(entity, self.tokens_src[idx_src : idx_src + depth])
            output = ' '.join(token.text for token in self.tokens_src[idx_src : idx_src + depth])
            logging.debug('Known multiword chain found: {output}')
            return True
        logging.debug('No multiword chain found')
        return False

    def _is_start_of_named_entity(self, idx_src: int) -> bool:
        '''Detect if token is a first word from named entity chain.'''
        return (
            self.tokens_src[idx_src].ent_iob_ == 'B'
            and len(self.tokens_src) > idx_src + 1
            and self.tokens_src[idx_src + 1].ent_iob_ == 'I'
        )

    def add_named_entity_to_trie(self, idx_src: int) -> None:
        '''Add named entity with it own translation.'''
        seq_tokens = [self.tokens_src[idx_src]]
        for forw_token in self.tokens_src[idx_src + 1 :]:
            if forw_token.ent_iob_ != 'I':
                break
            seq_tokens.append(forw_token)

        translation = self.container.translator.translate(' '.join(token.text for token in seq_tokens))
        entity = self.container.lemma_trie.add([token.lemma_ for token in seq_tokens], Entity(translation))
        self.treat_trie_entity(entity, seq_tokens)
        logging.debug(f'Named entity found and processed: {seq_tokens}')

    def treat_dict_entity(self, tokens_src: list['Token'], tokens_trg: list['Token']) -> None:
        '''Check and update repeat distance for both LemmaDict and it Entity child.'''
        lemma_src = ' '.join(token.lemma_ for token in tokens_src)
        lemma_trg = ' '.join(token.lemma_ for token in tokens_trg)
        text_src = ' '.join(token.text.lower() for token in tokens_src)
        text_trg = ' '.join(token.text.lower() for token in tokens_trg)
        lemma, entity = self.container.lemma_dict.add(lemma_src, text_src, lemma_trg, text_trg)

        if lemma.check_repeat(self.entity_counter) and entity.check_repeat(self.entity_counter):
            lemma.update(self.entity_counter)
            entity.update(self.entity_counter)
            self.result.append((' '.join(token.text.lower() for token in tokens_src), entity.translation))
            self.mfa_aligner.append_mfa_audio_to_output(tokens_src, tokens_trg)
        self.skip |= set(tokens_src)

    def treat_trie_entity(self, entity: Entity, tokens_src: list['Token'], tokens_trg=None) -> None:
        '''Check and update repeat distance for Entity (as LemmaTrie leaf).'''
        if entity.check_repeat(self.entity_counter):
            entity.update(self.entity_counter)
            self.result.append((' '.join(token.text.lower() for token in tokens_src), entity.translation))
            self.mfa_aligner.append_mfa_audio_to_output(tokens_src, tokens_trg if tokens_trg else entity.translation)
        self.skip |= set(tokens_src)

    @property
    def _translation_line_condition(self) -> bool:
        '''Rule for adding a complete sentence translation to result.'''
        n = len(self.result)
        quarter = len(self.tokens_src) // 4
        return n > 5 or 2 < n > quarter

    def get_results(self) -> list[tuple[int, str | list[tuple[str, str]]]]:
        '''Return original and translated (if needed) sentences [0, 1], vocabulary tokens[2] and whitespace tail[3].'''
        stripped = self.sentence.rstrip()
        tail = self.sentence[len(stripped) :]
        result = [(0, stripped)]
        if self.result:
            result.append((2, self.result))
            if self._translation_line_condition:
                result.append((1, self.translated_sentence))
        if tail:
            result.append((3, tail))
        return result

    def get_rendered_ssml(self) -> str:
        '''Return rendered ssml output for synthesis with provider that supports ssml.'''
        if not config.use_ssml:
            return ''
        translated = self.translated_sentence.rstrip() if self._translation_line_condition else ''
        return self.template.render(
            sentence=self.sentence.rstrip(),
            translated_sentence=translated,
            result=self.result,
            voice_trg=config.voice_trg,
            sentence_speed=config.sentence_pronunciation_speed,
            vocabulary_speed=config.vocabulary_pronunciation_speed,
        )

    def get_result_mfa_audio(self) -> 'AudioSegment':
        return self.mfa_aligner.get_result_audio(self._translation_line_condition)
