from collections import defaultdict
from itertools import accumulate, chain, pairwise
from math import ceil

import config
import dependencies as container
from moviepy import (AudioFileClip, ColorClip, CompositeVideoClip, ImageClip,
                     TextClip, concatenate_videoclips, vfx)

from core._structures import UserToken


class Video:
    def __init__(self) -> None:
        self.clips = []
        self.total_duration = 0

        try:
            self.background = (
                ImageClip(config.root_dir / 'background.jpg')
                .resized(width=config.video_width)
                .with_duration(self.total_duration)
            )
        except FileNotFoundError:
            self.background = ColorClip(
                size=(config.video_width, int(config.video_width / 16 * 9)), color=(66, 66, 66)
            ).with_duration(self.total_duration)

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

    def generate_subtitles(self, sentences: list['Sentence']) -> None:
        def format_time(seconds: float) -> str:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f'{h:01}:{m:02}:{s:05.2f}'

        def get_highlighted_textline(idxs, start_time) -> str:
            if config.highlight_after_main_sentence and start_time < min_start + (len(sentence.src_audio) // 1000):
                text = ''.join(
                    f'{{\\c&H{config.sub_colors[1]}&}}{text}{{\\c&H{config.sub_colors[0]}&}}' if i in idxs else text
                    for i, text in enumerate(token_texts)
                )
            else:
                text = ''.join(
                    f'{{\\c&H{config.sub_colors[1]}&}}{text}{{\\c&H{config.sub_colors[0]}&}}'
                    if i in idxs
                    else f'{{\\c&H{config.sub_colors[2]}&}}{text}{{\\c&H{config.sub_colors[0]}&}}'
                    if i in sentence.known_src_tokens and config.sub_colors[2]
                    else f'{{\\c&H{config.sub_colors[3]}&}}{text}{{\\c&H{config.sub_colors[0]}&}}'
                    if i in sentence.untranslatable_src_tokens and config.sub_colors[3]
                    else text
                    for i, text in enumerate(token_texts)
                )
            return text.strip().replace('\n', '')

        result = []
        for sentence in sentences:
            self.collect_segments(sentence)
            segments = defaultdict(set)
            token_texts = []
            glob_i = 0
            min_start = float('inf')
            for b, tokens in enumerate((sentence.src_tokens, sentence.trg_tokens)):
                for token in tokens:
                    for start, _ in token.segments:
                        min_start = min(min_start, start)
                        segments[start].add(glob_i)
                    token_texts.append(f'{token.text} ' if token.whitespace else token.text)
                    glob_i += 1
                if not b:
                    token_texts.append('{\\fs%s}\\N\\N{\\fs%s}' % (config.font_size // 2, config.font_size))
                    glob_i += 1

            result.extend(
                (start_time, get_highlighted_textline(idxs, start_time)) for start_time, idxs in segments.items()
            )

        result = [
            (format_time(start), format_time(end), text)
            for (start, text), (end, _) in pairwise(sorted(result + [(359880.0, '')]))
        ]
        rendered = container.templates['subtitles'].render(
            {
                'result': result,
                'width': self.background.w,
                'height': self.background.h,
                'font_name': config.caption_font,
                'font_size': config.font_size,
                'bottom_margin': config.bottom_margin,
            }
        )
        return rendered

    def append_sentence_to_clips(self, sentence: 'Sentence') -> None:
        '''Create clips with static words and highlighted versions and append them to list of clips.'''
        start_of_sentence = self.total_duration
        self.collect_segments(sentence)

        src_lines = self.get_lines(sentence.src_text, sentence.src_tokens)
        trg_lines = self.get_lines(sentence.trg_text, sentence.trg_tokens)
        curr_h = self.background.h - config.bottom_margin - config.margin_between_original_and_translation
        curr_h -= config.line_height * (len(src_lines) + len(trg_lines))
        widths = []

        for lines in (src_lines, trg_lines):
            for tokens in lines:  # for line in lines
                line_w = self.create_text_clip(f'{" ".join(word.text for word in tokens)}', 0, 0, 0).w
                widths.append([(config.video_width - line_w) // 2])
                for word in tokens:
                    text = f'{word.text} ' if word.whitespace else word.text
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
            curr_duration = self.total_duration
            for token in tokens:
                if not token.audio:
                    continue
                token_dur = (token.audio.stop - token.audio.start - config.start_shift_ms - config.end_shift_ms) / 1000
                token.segments.append((curr_duration, token_dur))
                curr_duration += token_dur

        s_break = config.break_between_sentences_ms / 1000
        vo_break = config.break_between_vocabulary_ms / 1000
        vi_break = config.break_in_vocabulary_ms / 1000

        apply_sentence_tokens(sentence.src_tokens)
        self.total_duration += len(sentence.src_audio) / 1000 + s_break

        for i, (src_tokens, trg_tokens, src_audio, trg_audio) in enumerate(sentence.vocabulary):
            curr_dur = (len(src_audio) + len(trg_audio)) / 1000 + vi_break + (vo_break if i else 0)
            for token in chain(src_tokens, trg_tokens):
                token.segments.append((self.total_duration, curr_dur))
            self.total_duration += curr_dur
        self.total_duration += s_break if sentence.vocabulary else 0

        if sentence.show_translation:
            apply_sentence_tokens(sentence.trg_tokens)
            self.total_duration += len(sentence.trg_audio) / 1000 + s_break

            if config.repeat_original_sentence_after_translated:
                apply_sentence_tokens(sentence.src_tokens)
                self.total_duration += len(sentence.src_audio) / 1000 + s_break

    def compose_clips(self) -> CompositeVideoClip:
        '''Final composition of all clips with background and audio (both from files).'''
        composite = CompositeVideoClip([self.background, *self.clips])
        audio_clip = AudioFileClip(config.root_dir / 'output_audio.wav')
        return composite.with_audio(audio_clip)

    def save_video(self, video: CompositeVideoClip) -> None:
        video.write_videofile(config.root_dir / 'output_video.mp4', fps=24)
