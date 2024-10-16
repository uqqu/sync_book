import logging

import regex as re

from _structures import Entity
from alignment import TokenAligner
from config import Config


class Sentence:
    def __init__(self, sentence: str, current_pos: int, container: 'DependencyContainer') -> None:
        self.sentence = sentence
        self.current_pos = current_pos
        self.container = container
        self.config = container.config
        self.translated_sentence = self.container.translator.translate(sentence)

        self.tokens_src = container.text_processor.get_merged_tokens(self.sentence, lang=self.config.source_lang)
        self.tokens_trg = container.text_processor.get_merged_tokens(
            self.translated_sentence, lang=self.config.target_lang
        )
        logging.info(f'{self.tokens_src=}, {self.tokens_trg=}')
        container.text_processor.add_tokens_embeddings(self.sentence, self.tokens_src)
        container.text_processor.add_tokens_embeddings(self.translated_sentence, self.tokens_trg)

        self.aligner = TokenAligner(container, self.tokens_src, self.tokens_trg)

        self.result: list[tuple[str, str]] = []
        self.possible_result: list[tuple[str, str]] = []
        self.skip: set['Token'] = set()

    def process_audio(self) -> None:
        '''Add original and translated audio and recognized from it segments to token attributes if mfa enabled.'''
        if not self.config.speech_synth:
            return
        sentence_audio = self.container.synthesizer.synthesize_fragment(self.sentence, self.config.source_lang)
        translated_audio = self.container.synthesizer.synthesize_fragment(
            self.translated_sentence, self.config.target_lang
        )
        self.output_audio = sentence_audio
        self.translated_audio = translated_audio
        if self.config.mfa_use:
            self._process_mfa(sentence_audio, translated_audio)

    def _process_mfa(self, sentence_audio: 'BytesIO', translated_audio: 'BytesIO') -> None:
        '''Recognize and set word fragments to token attributes for future addition to audio output.'''
        synth = self.container.synthesizer
        alignment_src = synth.align_audio(self.sentence, sentence_audio, self.config.source_full_lang, 'temp/temp_src')
        segments_src = synth.split_audio_by_alignment(sentence_audio, alignment_src)
        alignment_trg = synth.align_audio(
            self.translated_sentence, translated_audio, self.config.target_full_lang, 'temp/temp_trg'
        )
        segments_trg = synth.split_audio_by_alignment(translated_audio, alignment_trg)
        for segments, tokens in ((segments_src, self.tokens_src), (segments_trg, self.tokens_trg)):
            i, j = 0, 0
            while i < len(segments) and j < len(tokens):
                if segments[i]['text'] in tokens[j].text.lower():
                    tokens[j]._.audio = synth.add(tokens[j]._.audio, segments[i]['audio'])
                    i += 1
                else:
                    j += 1

    def _append_mfa_audio_to_output(self, tokens_src: list['Token'], target: list['Token'] | str) -> None:
        '''Add recognized word fragments from resulting token attributes to sentence audio output.'''
        if not self.config.speech_synth or not self.config.mfa_use:
            return
        synth = self.container.synthesizer
        for token in tokens_src:
            self.output_audio = synth.add(self.output_audio, token._.audio)
        if isinstance(target, list):
            for token in target:
                self.output_audio = synth.add(self.output_audio, token._.audio)
        else:
            translation_audio = synth.synthesize_fragment(target, self.config.target_lang)
            self.output_audio = synth.add(self.output_audio, translation_audio)

    def process_tokens(self) -> None:
        for idx_src, token_src in enumerate(self.tokens_src):
            # TODO POS aligning
            self.current_pos += 1
            logging.debug(f'Processing token: {token_src}')
            if token_src in self.skip:  # skip token if it has already been processed
                logging.debug('Skipping previously processed token')
            elif not re.match(self.config.word_pattern, token_src.text):  # skip punct w/o counting
                self.current_pos -= 1
                logging.debug('Skipping punctuation')
            elif token_src.ent_type_ in self.config.untranslatable_entities:
                logging.debug(f'Skipping untranslatable entity: {token_src.ent_type_}')
            elif self.trie_search(idx_src):  # check multiword origin chain
                logging.debug('Found multiword chain')
            elif self._is_start_of_named_entity(idx_src):  # treat named entity chain and add tokens to 'skip'
                self.add_named_entity_to_trie(idx_src)
            elif token_src.text.lower() in self.config.stop_words:  # only after multiword check
                logging.debug('Skipping stopword')
            else:
                score, seq_tokens_src, seq_tokens_trg = self.aligner.process_alignment(idx_src)
                if score < self.config.min_align_weight:
                    logging.debug(f'Rejected after alignment: {score}, {seq_tokens_src}, {seq_tokens_trg}')
                    continue
                if len(seq_tokens_src) == 1:
                    self.treat_dict_entity(idx_src, seq_tokens_src, seq_tokens_trg)
                else:
                    translation = ' '.join(token.text for token in seq_tokens_trg)
                    self.treat_trie_entity(Entity(translation), idx_src, seq_tokens_src, seq_tokens_trg)

        logging.info(f'Result: {self.result}, Possible Result: {self.possible_result}')

    def trie_search(self, idx_src: int) -> bool:
        '''Look for existing source multiword sequence in LemmaTrie.'''
        entity, depth = self.container.lemma_trie.search(self.tokens_src[idx_src:])
        if depth > 1:
            self.treat_trie_entity(entity, idx_src, self.tokens_src[idx_src : idx_src + depth])
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
        self.treat_trie_entity(entity, idx_src, seq_tokens)
        logging.debug(f'Named entity found and processed: {seq_tokens}')

    def treat_dict_entity(self, idx: int, tokens_src: list['Token'], tokens_trg: list['Token']) -> None:
        '''Check and update repeat distance for both LemmaDict and it Entity child.'''
        lemma_src = ' '.join(token.lemma_ for token in tokens_src)
        lemma_trg = ' '.join(token.lemma_ for token in tokens_trg)
        text_src = ' '.join(token.text.lower() for token in tokens_src)
        text_trg = ' '.join(token.text.lower() for token in tokens_trg)
        lemma, entity = self.container.lemma_dict.add(lemma_src, text_src, lemma_trg, text_trg)

        pos = self.current_pos + idx
        if lemma.check_repeat(pos) and entity.check_repeat(pos):
            lemma.update(pos)
            entity.update(pos)
            self.result.append((' '.join(token.text for token in tokens_src), entity.translation))
            self._append_mfa_audio_to_output(tokens_src, entity.translation)
        self.skip |= set(tokens_src)

    def treat_trie_entity(self, entity: Entity, idx: int, tokens_src: list['Token'], tokens_trg=None) -> None:
        '''Check and update repeat distance for Entity (as LemmaTrie leaf).'''
        pos = self.current_pos + idx
        if entity.check_repeat(pos):
            entity.update(pos)
            self.result.append((' '.join(token.text.lower() for token in tokens_src), entity.translation))
            self._append_mfa_audio_to_output(tokens_src, tokens_trg)
        self.skip |= set(tokens_src)

    def _translation_line_condition(self) -> bool:
        '''Rule for adding a complete sentence translation to result.'''
        n = len(self.result)
        third = len(self.tokens_src) // 3
        return n > 6 or 2 < n > third

    def get_result_text(self) -> str:
        '''Return combined text: original sentence, resulting tokens and translated sentence if needed.'''
        result = self.sentence
        delim = self.config.lang_delimiter
        if self.result:
            result += ' [{}] '.format(', '.join(f'{src}–{delim}{trg}{delim}' for src, trg in self.result))
            # TODO add possible result?
            if self._translation_line_condition():
                result += f'{delim}{self.translated_sentence}{delim} '

        return result

    def get_result_audio(self) -> 'AudioSegment':
        '''Return combined audio: original sentence, resulting tokens and translated sentence if needed.'''
        synth = self.container.synthesizer
        if not self.config.mfa_use:
            for res_src, res_trg in self.result:
                audio_src = synth.synthesize_fragment(res_src, self.config.source_lang)
                audio_trg = synth.synthesize_fragment(res_trg, self.config.target_lang)
                self.output_audio = synth.add(self.output_audio, synth.silent(50), audio_src, audio_trg)
        if self._translation_line_condition():
            self.output_audio = synth.add(self.output_audio, self.translated_audio)
        return self.output_audio
