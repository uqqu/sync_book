﻿import logging
import os
import subprocess

import tgt

import config


class MFAligner:
    def __init__(self, parent: 'Sentence') -> None:
        if not config.speech_synth or not config.use_mfa:
            self.idle = True
            return
        self.idle = False
        self.parent = parent
        self.synth = parent.container.synthesizer
        speed = config.sentence_pronunciation_speed
        self.sentence_audio = self.synth.synthesize(self.parent.sentence, config.source_lang, speed)
        self.translated_audio = self.synth.synthesize(self.parent.translated_sentence, config.target_lang, speed)
        self.output_audio = self.sentence_audio[:]
        self._process_mfa_alignment()

    def _process_mfa_alignment(self) -> None:
        '''Recognize and set word fragments to token attributes for future addition to audio output.'''
        alignment_src = self._align_audio(
            self.parent.sentence, self.sentence_audio, config.source_full_lang, 'temp/temp'
        )
        segments_src = self._split_audio_by_alignment(self.sentence_audio, alignment_src)
        alignment_trg = self._align_audio(
            self.parent.translated_sentence, self.translated_audio, config.target_full_lang, 'temp/temp'
        )
        segments_trg = self._split_audio_by_alignment(self.translated_audio, alignment_trg)
        for segments, tokens in ((segments_src, self.parent.tokens_src), (segments_trg, self.parent.tokens_trg)):
            i, j = 0, 0
            while i < len(segments) and j < len(tokens):
                if segments[i]['text'] in tokens[j].text.lower():
                    tokens[j]._.audio += segments[i]['audio']
                    i += 1
                else:
                    j += 1

    def append_mfa_audio_to_output(self, result_src: list['Token'], result_trg: list['Token'] | str) -> None:
        '''Add recognized word fragments from resulting token attributes to sentence audio output.'''
        if self.idle:
            return
        for token in result_src:
            self.output_audio += token._.audio
        self.output_audio += self.synth.silent(200)
        if isinstance(result_trg, list):
            for token in result_trg:
                self.output_audio += token._.audio
        else:
            translation_audio = self.synth.synthesize(result_trg, config.target_lang)
            self.output_audio += translation_audio

    def _align_audio(self, text: str, audio_buffer: 'AudioSegment', lang: str, output_pathfile: str) -> str:
        '''Align provided audio buffer with the given text using MFA.'''
        os.makedirs('temp', exist_ok=True)
        audio_buffer.export('temp/temp.wav', format='wav')
        with open('temp/temp.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        dict_path = f'{config.mfa_dir}dictionary//{lang}_mfa.dict'
        model_path = f'{config.mfa_dir}acoustic//{lang}_mfa.zip'
        command = ['mfa', 'align', '--clean', '--single_speaker', 'temp', dict_path, model_path, './temp/']
        subprocess.run(command, check=True)
        return f'{output_pathfile}.TextGrid'

    def _split_audio_by_alignment(self, audio: 'AudioSegment', textgrid_pathfile: str) -> list[dict]:
        '''Split the given audio buffer into segments based on alignment data from a TextGrid file.'''
        textgrid = tgt.io.read_textgrid(textgrid_pathfile)
        segments = []

        word_tier = textgrid.get_tier_by_name('words')
        for interval in word_tier.intervals:
            if interval.text.strip():
                start_time = max(0, interval.start_time * 1000 - config.mfa_start_shift_ms)
                end_time = min(len(audio), interval.end_time * 1000 + config.mfa_end_shift_ms)
                segments.append({'text': interval.text, 'audio': audio[start_time:end_time]})

        return segments

    def get_result_audio(self, additional_translation: bool = False) -> 'AudioSegment':
        if additional_translation:
            self.output_audio += self.translated_audio
        return self.output_audio