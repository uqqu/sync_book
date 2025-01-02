import string
from itertools import pairwise

import config
import dependencies as container
from pydub import AudioSegment

from core._structures import Entity, UserToken


class Sentence:
    def __init__(self, original: str, translated: str, tokens: tuple[list, ...], show_translation=None) -> None:
        self.src_text = original
        self.trg_text = translated

        self.src_tokens: list[UserToken] = tokens[0]
        self.trg_tokens: list[UserToken] = tokens[1]
        self.result: (
            list[tuple[list[UserToken], list[UserToken], bool | None]] | list[tuple[list[UserToken], list[UserToken]]]
        ) = tokens[2]

        self.show_translation: bool = (
            self._translation_line_condition() if show_translation is None else show_translation
        )

    def update_positions(self) -> None:
        '''Apply new positions to user structure after text processing or draft load step.'''
        # cleanup rejected result pairs and simplify types
        self.result = [res[:2] for res in self.result if res[2]]  # type: ignore
        for src_res, trg_res in self.result:
            pos = src_res[0].position
            if len(src_res) > 1:  # trie
                entity, _ = container.structures.lemma_trie.search(src_res)
                if entity is None:
                    # when a new path was loaded from the draft.json
                    entity = container.structures.lemma_trie.add(src_res, Entity(' '.join(t.text for t in trg_res)))
                entity.update(pos)
                continue

            lemma, entity = container.structures.lemma_dict.add(src_res, trg_res)
            if lemma.check_repeat(pos) and entity.check_repeat(pos):
                lemma.update(pos)
                entity.update(pos)

    def gen_62(self, start: int = 0):
        '''Convert int to 62-base with saving previous value.'''
        i = start
        alphabet = f'{string.digits}{string.ascii_letters}'
        while True:
            num = i
            if num == 0:
                i += 1
                yield '0'
                continue
            result = []
            while num > 0:
                num, rem = divmod(num, 62)
                result.append(alphabet[rem])
            yield ''.join(reversed(result))
            i += 1

    def get_full_ssml_config(self, idx=0) -> tuple[dict, int]:
        '''Get the necessary dict values to render sentence with ssml jinja template. Add last <mark> pos value.'''
        src_sentence = self.get_sentence_ssml_config(True, idx)
        idx += len(self.src_tokens)
        result = self.get_vocabulary_ssml_config(idx)
        idx += len(self.result) * 2
        trg_sentence = self.get_sentence_ssml_config(False, idx)
        idx += len(self.trg_tokens)
        additional = {
            'sent_break': config.break_between_sentences_ms,
            'repeat_original': config.repeat_original_sentence_after_translated,
        }
        if config.repeat_original_sentence_after_translated:
            additional |= {f'{k}_extra': v for k, v in self.get_sentence_ssml_config(True, idx).items()}
            idx += len(self.src_tokens)
        return (src_sentence | trg_sentence | result | additional), idx

    def get_sentence_ssml_config(self, is_src: bool, idx=0) -> dict:
        '''Get the necessary dict values to render sentence parts (src_text, trg_text) with ssml jinja template.'''
        tokens = self.src_tokens if is_src else self.trg_tokens
        dummy = UserToken(
            text='', lemma_=None, index=tokens[-1].index + len(tokens[-1].text), position=None, is_punct=True
        )
        words = [f'{a.text}{b.text}' if b.is_punct else a.text for a, b in pairwise(tokens + [dummy]) if not a.is_punct]
        if config.use_ssml == 2:
            gen = self.gen_62(idx)
            words = [f'<mark name="{next(gen)}"/>{word}' for word in words]
        speed = config.sentence_pronunciation_speed
        if is_src:
            return {'sentence_speed': speed, 'src_sentence': words}
        return {'sentence_speed': speed, 'trg_sentence': words, 'trg_voice': config.target_voice}

    def get_vocabulary_ssml_config(self, idx=0) -> dict:
        '''Get the necessary dict values to render vocabulary with ssml jinja template.'''
        if config.use_ssml == 2:
            result = []
            gen = self.gen_62(idx)
            joined_with_mark = lambda tokens: f'<mark name="{next(gen)}"/>{" ".join(token.text for token in tokens)}'
            for src_res, trg_res in self.result:  # type: ignore
                result.append((joined_with_mark(src_res), joined_with_mark(trg_res)))
        else:
            result = [(' '.join(s.text for s in src), ' '.join(t.text for t in trg)) for src, trg in self.result]

        res = {
            'vocabulary_speed': config.vocabulary_pronunciation_speed,
            'vocab_inner_break': config.break_in_vocabulary_ms,
            'vocab_outer_break': config.break_between_vocabulary_ms,
            'trg_voice': config.target_voice,
            'pitch': config.ssml_vocabulary_pitch,
            'volume': config.ssml_vocabulary_volume,
            'result': result,
        }
        return res

    def synthesize_sentence(self) -> None:
        def approximately_arrange_audio_tokens(tokens: list[UserToken], sent_audio: AudioSegment) -> None:
            '''Approximately set audio slice for unspecified tokens.'''
            # TODO? can be improved
            denom = sum(len(token.text) for token in tokens) * (len(sent_audio) - config.final_silence_of_sentences_ms)
            prev = 0
            for token in tokens:
                token.audio = slice(prev, prev + len(token.text) / denom)
                prev = token.audio.stop

        src_words = [t for t in self.src_tokens if not t.is_punct]
        trg_words = [t for t in self.trg_tokens if not t.is_punct]
        result = container.synthesizer.synthesize_sentence(self)
        if isinstance(result[0], AudioSegment):
            (self.src_audio, self.trg_audio) = result
            approximately_arrange_audio_tokens(src_words, self.src_audio)
            approximately_arrange_audio_tokens(trg_words, self.trg_audio)
            return

        (self.src_audio, src_ts), (self.trg_audio, trg_ts) = result
        for words, ts, main_audio in ((src_words, src_ts, self.src_audio), (trg_words, trg_ts, self.trg_audio)):
            for token, (start, end) in zip(words, pairwise(ts)):
                token.audio = slice(start - config.start_shift_ms, end + config.end_shift_ms)
            words[-1].audio = slice(end - config.start_shift_ms, len(main_audio) - config.final_silence_of_sentences_ms)

    def _translation_line_condition(self) -> bool:
        '''Rule for adding a complete sentence translation to result.'''
        if not config.min_part_of_new_words_to_add:
            return False

        n = len(self.result)
        return (
            n >= config.min_count_of_new_words_to_add
            and n >= len(self.src_tokens) // config.min_part_of_new_words_to_add
        )
