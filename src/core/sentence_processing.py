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
        self.vocabulary: (
            list[tuple[list[UserToken], list[UserToken], bool | None]] | list[tuple[list[UserToken], list[UserToken]]]
        ) = tokens[2]
        self.known_src_tokens = tokens[3]
        self.untranslatable_src_tokens = tokens[4]

        self.show_translation: bool = (
            self._translation_line_condition() if show_translation is None else show_translation
        )

    def update_positions(self) -> None:
        '''Apply new positions to user structure after text processing or draft load; and combine token sequences.'''
        next_src_idx = {a.index: b.index for a, b in pairwise(self.src_tokens)}
        next_trg_idx = {a.index: b.index for a, b in pairwise(self.trg_tokens)} | {None: -1}
        groups = []
        prev_group = ()
        for src_voc, trg_voc, status in self.vocabulary:
            if not status:
                continue

            pos = src_voc[0].position
            if len(src_voc) > 1:  # trie
                entity, _ = container.structures.lemma_trie.search(src_voc)
                if entity is None:
                    # when a new path was loaded from the draft.json
                    entity = container.structures.lemma_trie.add(src_voc, Entity(' '.join(t.text for t in trg_voc)))
                entity.update(pos)
            else:
                lemma, entity = container.structures.lemma_dict.add(src_voc, trg_voc)
                if lemma.check_repeat(pos) and entity.check_repeat(pos):
                    lemma.update(pos)
                    entity.update(pos)

            if prev_group:
                if (
                    src_voc[-1].index == next_src_idx[prev_group[0][-1].index]
                    and trg_voc[-1].index == next_trg_idx[prev_group[1][-1].index]
                ):
                    prev_group[0].extend(src_voc)
                    prev_group[1].extend(trg_voc)
                    continue
                else:
                    groups.append(prev_group)
            prev_group = (src_voc, trg_voc)
        if prev_group:
            groups.append(prev_group)
        self.vocabulary = groups

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

    def get_sentence_ssml_config(self, is_src: bool, idx=0) -> dict:
        '''Get the necessary dict values to render sentence parts (src_text, trg_text) with ssml jinja template.'''
        tokens = self.src_tokens if is_src else self.trg_tokens
        words = [f'{t.text} ' if t.whitespace else t.text for t in tokens if not t.is_punct]
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
            vocab = []
            gen = self.gen_62(idx)
            joined_with_mark = lambda tokens: f'<mark name="{next(gen)}"/>{" ".join(token.text for token in tokens)}'
            for src_voc, trg_voc in self.vocabulary:  # type: ignore
                vocab.append((joined_with_mark(src_voc), joined_with_mark(trg_voc)))
        else:
            vocab = [(' '.join(s.text for s in src), ' '.join(t.text for t in trg)) for src, trg in self.vocabulary]

        res = {
            'vocabulary_speed': config.vocabulary_pronunciation_speed,
            'vocab_inner_break': config.break_in_vocabulary_ms,
            'vocab_outer_break': config.break_between_vocabulary_ms,
            'trg_voice': config.target_voice,
            'pitch': config.ssml_vocabulary_pitch,
            'volume': config.ssml_vocabulary_volume,
            'vocabulary': vocab,
        }
        return res

    def synthesize_sentence(self) -> None:
        def approximately_arrange_audio_tokens(tokens: list[UserToken], sent_audio: AudioSegment) -> None:
            '''Approximately set audio slice for unspecified tokens.'''
            # TODO? can be improved
            if not tokens:
                return
            prev = 0
            ms_per_symb = (len(sent_audio) - config.final_silence_of_sentences_ms) / sum(
                len(token.text) for token in tokens
            )
            for token in tokens:
                token.audio = slice(prev, prev + ms_per_symb * len(token.text))
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
            if not words:
                break
            if len(ts) == 1:
                words[0].audio = slice(0, len(main_audio) - config.final_silence_of_sentences_ms + config.end_shift_ms)
                continue
            for token, (start, end) in zip(words, pairwise(ts)):
                token.audio = slice(start - config.start_shift_ms, end + config.end_shift_ms)
            words[-1].audio = slice(
                end - config.start_shift_ms,
                len(main_audio) - config.final_silence_of_sentences_ms + config.end_shift_ms,
            )

    def synthesize_vocabulary(self) -> None:
        '''Add separated vocabulary audio (src, trg) to every pair in sentence.vocabulary.'''

        def collect_vocabulary_audio(tokens: list[UserToken], sent_audio: AudioSegment, lang: str) -> AudioSegment:
            '''Compose vocabulary audio from prerecognized segments from sentence audio if possible.'''
            synth = container.synthesizer
            if not config.reuse_synthesized or any(token.audio is None for token in tokens):
                audio = synth.synthesize(' '.join(s.text for s in tokens), lang, ssml=False)
                result_audio = synth.adjust_audio_speed(audio, config.vocabulary_pronunciation_speed)
                if len(tokens) == 1 and not tokens[0].audio:
                    tokens[0].audio = len(result_audio)
            else:
                result_audio = sent_audio[tokens[0].audio.start : tokens[-1].audio.stop]
                if (speed := config.vocabulary_pronunciation_speed / config.sentence_pronunciation_speed) < 0.5:
                    result_audio = synth.adjust_audio_speed(result_audio, 1 / config.sentence_pronunciation_speed)
                    result_audio = synth.adjust_audio_speed(result_audio, config.vocabulary_pronunciation_speed)
                else:
                    result_audio = synth.adjust_audio_speed(result_audio, speed)
            crossfade = config.crossfade_ms
            while crossfade > len(result_audio):
                crossfade >>= 1
            result_audio = result_audio.fade_in(crossfade).fade_out(crossfade)

            return result_audio

        if config.use_ssml == 2 and not config.reuse_synthesized:
            # full vocabulary at once
            ssml_config = self.get_vocabulary_ssml_config()
            voc_ssml = container.templates['vocabulary'].render(ssml_config)
            voc_audio, voc_ts = container.synthesizer.synthesize(f'<speak>{voc_ssml}</speak>', with_timestamps=True)
            prev = None
            for i, (start, stop) in enumerate(pairwise(voc_ts + [len(voc_audio)])):
                if i % 2:
                    self.vocabulary[i // 2] = (*self.vocabulary[i // 2], voc_audio[prev], voc_audio[start:stop])
                else:
                    prev = slice(start, stop)
        elif config.use_ssml:
            for i, (src_voc_tokens, trg_voc_tokens) in enumerate(self.vocabulary[:]):
                audio = []
                for tokens, main_audio, lang, voice in (
                    (src_voc_tokens, self.src_audio, config.source_lang, ''),
                    (trg_voc_tokens, self.trg_audio, config.target_lang, config.target_voice),
                ):
                    if config.reuse_synthesized:
                        audio.append(collect_vocabulary_audio(tokens, main_audio, lang))
                        continue

                    voc_ssml = container.templates['single_vocabulary_item'].render(
                        vocabulary_speed=config.vocabulary_pronunciation_speed,
                        pitch=config.ssml_vocabulary_pitch,
                        volume=config.ssml_vocabulary_volume,
                        voc_text=' '.join(t.text for t in tokens),
                        voice=voice,
                    )
                    audio.append(container.synthesizer.synthesize(f'<speak>{voc_ssml}</speak>'))
                self.vocabulary[i] = (*self.vocabulary[i], *audio)
        else:
            for i, (src_voc_tokens, trg_voc_tokens) in enumerate(self.vocabulary[:]):
                src_voc_audio = collect_vocabulary_audio(src_voc_tokens, self.src_audio, config.source_lang)
                trg_voc_audio = collect_vocabulary_audio(trg_voc_tokens, self.trg_audio, config.target_lang)
                self.vocabulary[i] = (*self.vocabulary[i], src_voc_audio, trg_voc_audio)

    def _translation_line_condition(self) -> bool:
        '''Rule for adding a complete sentence translation to result.'''
        if not config.min_part_of_new_words_to_add:
            return False

        n = sum(item[2] for item in self.vocabulary)
        return (
            n >= config.min_count_of_new_words_to_add
            and n >= len(self.src_tokens) // config.min_part_of_new_words_to_add
        )
