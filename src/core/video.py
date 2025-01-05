from collections import defaultdict
from itertools import accumulate, chain, pairwise
from math import ceil

import config
from moviepy import (AudioFileClip, ColorClip, CompositeVideoClip, ImageClip,
                     TextClip, concatenate_videoclips, vfx)

from core._structures import UserToken


class Video:
    def __init__(self) -> None:
        self.clips = []
        self.total_duration = 0

    @staticmethod
    def create_text_clip(text: str, duration: float, w: int, h: int, color='white') -> TextClip:
        return (
            TextClip(
                font=config.caption_font,
                text=text,
                font_size=config.font_size,
                color=color,
                size=(None, 100),
                stroke_color='black',
                stroke_width=2,
            )
            .with_duration(duration)
            .with_effects([vfx.CrossFadeIn(0.1), vfx.CrossFadeOut(0.1)])
            .with_position((w, h))
        )

    def append_sentence_to_clips(self, sentence: 'Sentence') -> None:
        '''Create clips with static words and highlighted versions and append them to list of clips.'''
        start_of_sentence = self.total_duration
        self.collect_segments(sentence)

        src_lines = self.get_lines(sentence.src_text, sentence.src_tokens)
        trg_lines = self.get_lines(sentence.trg_text, sentence.trg_tokens)
        curr_h = config.video_height - config.bottom_margin - config.margin_between_original_and_translation
        curr_h -= config.line_height * (len(src_lines) + len(trg_lines))
        widths = []
        dummy = UserToken(text='', lemma_='', index=0, position=0, is_punct=True)

        for lines in (src_lines, trg_lines):
            for tokens in lines:  # for line in lines
                line_w = self.create_text_clip(f'{" ".join(word.text for word in tokens)}', 0, 0, 0).w
                widths.append([(config.video_width - line_w) // 2])
                for word, next_word in pairwise(tokens + [dummy]):
                    text = f'{word.text}{" " if not next_word.is_punct else ""}'
                    clip = self.create_text_clip(text, self.total_duration - start_of_sentence, widths[-1][-1], curr_h)
                    self.clips.append(clip.with_start(start_of_sentence))
                    for start, duration in word.segments:
                        if not duration:
                            continue
                        clip = self.create_text_clip(text, duration, widths[-1][-1], curr_h, 'yellow')
                        self.clips.append(clip.with_start(start))

                    widths[-1].append(widths[-1][-1] + self.clips[-1].w)
                curr_h += config.line_height

            curr_h += config.margin_between_original_and_translation

    def get_lines(self, text: str, tokens: list[UserToken]) -> list[list[UserToken]]:
        '''Split a list of tokens into lines to fit them within a given width.'''
        width = self.create_text_clip(text, 0, 0, 0).w
        num_lines = ceil(width / config.video_width)
        lines = [[] for _ in range(num_lines)]

        total_length = len(text)
        target_length = total_length / num_lines

        current_lengths = [0] * num_lines
        line_index = 0

        for token in tokens:
            new_length = current_lengths[line_index] + len(token.text) + (1 if lines[line_index] else 0)
            if new_length > target_length and line_index < num_lines - 1:
                line_index += 1
            lines[line_index].append(token)
            current_lengths[line_index] += len(token.text) + (1 if lines[line_index] else 0)
        return lines

    def collect_segments(self, sentence: 'Sentence') -> None:
        '''Identify the segments of the original audio in which every word is spoken, in order to highlight them.'''

        def apply_sentence_tokens(tokens: list[UserToken]) -> None:
            '''Helper for basic segmentation of the full sentence.'''
            for token in tokens:
                if not token.audio:
                    continue
                curr_dur = (token.audio.stop - token.audio.start - config.start_shift_ms - config.end_shift_ms) / 1000
                token.segments.append((self.total_duration, curr_dur))
                self.total_duration += curr_dur
            self.total_duration += s_break

        s_break = config.break_between_sentences_ms / 1000
        vo_break = config.break_between_vocabulary_ms / 1000
        vi_break = config.break_in_vocabulary_ms / 1000
        voc_speed = config.vocabulary_pronunciation_speed / config.sentence_pronunciation_speed * 1000  # with ms->s

        self.total_duration += sentence.src_tokens[0].audio.start / 1000 if sentence.src_tokens[0].audio else 0

        apply_sentence_tokens(sentence.src_tokens)

        for src_tokens, trg_tokens, src_audio, trg_audio in sentence.vocabulary:
            curr_dur = (len(src_audio) + len(trg_audio)) / 1000 + vi_break + vo_break
            for token in chain(src_tokens, trg_tokens):
                token.segments.append((self.total_duration, curr_dur))
            self.total_duration += curr_dur
        self.total_duration += s_break - vo_break

        if sentence.show_translation:
            self.total_duration += sentence.trg_tokens[0].audio.start / 1000 if sentence.trg_tokens[0].audio else 0
            apply_sentence_tokens(sentence.trg_tokens)

            if config.repeat_original_sentence_after_translated:
                self.total_duration += sentence.src_tokens[0].audio.start / 1000 if sentence.src_tokens[0].audio else 0
                apply_sentence_tokens(sentence.src_tokens)

    def compose_clips(self) -> CompositeVideoClip:
        '''Final composition of all clips with background and audio (both from files).'''
        try:
            background = (
                ImageClip(config.root_dir / 'background.jpg')
                .resized(width=config.video_width, height=config.video_height)
                .with_duration(self.total_duration)
            )
        except OSError:
            background = ColorClip(size=(config.video_width, config.video_height), color=(66, 66, 66)).with_duration(
                self.total_duration
            )
        composite = CompositeVideoClip([background, *self.clips])
        audio_clip = AudioFileClip(config.root_dir / 'multilingual_output.wav')
        return composite.with_audio(audio_clip)

    def save_video(self, video: CompositeVideoClip, name: str) -> None:
        video.write_videofile(config.root_dir / f'{name}.mp4', fps=24)
