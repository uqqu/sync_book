import logging
import os
import subprocess
from io import BytesIO

import regex as re
import tgt
from gtts import gTTS
from pydub import AudioSegment


class SpeechSynthesizer:
    def __init__(self, config: 'Config') -> None:
        self.config = config

    @staticmethod
    def silent(duration: int) -> AudioSegment:
        return AudioSegment.silent(duration=duration)

    def add(self, *audio_segments: AudioSegment | BytesIO) -> AudioSegment:
        '''Combine multiple audio segments into a single segment.'''
        result_audio = AudioSegment.silent(0)
        for segment in audio_segments:
            if not isinstance(segment, AudioSegment):
                segment = AudioSegment.from_wav(segment)
            result_audio += segment
        return result_audio

    def synthesize_fragment(self, text: str, lang: str) -> BytesIO:
        '''Generate speech synthesis for provided text using gTTS.'''
        text = re.sub(r'^\P{L}+|[\P{L}?!\.]+$', '', text)
        audio_buffer = BytesIO()
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer

    def align_audio(self, text: str, audio_buffer: BytesIO, lang: str, output_pathfile: str) -> str:
        '''Align provided audio buffer with the given text using MFA.'''
        audio_buffer.seek(0)
        os.makedirs('temp', exist_ok=True)
        with open('temp/temp.wav', 'wb') as f:
            f.write(audio_buffer.getbuffer())
        with open('temp/temp.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        dict_path = f'{self.config.mfa_dir}dictionary//{lang}_mfa.dict'
        model_path = f'{self.config.mfa_dir}acoustic//{lang}_mfa.zip'
        command = ['mfa', 'align', '--clean', '--single_speaker', 'temp', dict_path, model_path, './temp/']
        subprocess.run(command, check=True)
        return f'{output_pathfile}.TextGrid'

    def split_audio_by_alignment(self, audio_buffer: BytesIO, textgrid_pathfile: str) -> list[dict]:
        '''Split the given audio buffer into segments based on alignment data from a TextGrid file.'''
        audio = AudioSegment.from_file(audio_buffer, format='wav')
        textgrid = tgt.io.read_textgrid(textgrid_pathfile)
        segments = []

        word_tier = textgrid.get_tier_by_name('words')
        for interval in word_tier.intervals:
            if interval.text.strip():
                start_time = max(0, interval.start_time * 1000 - self.config.mfa_start_shift)
                end_time = min(len(audio), interval.end_time * 1000 + self.config.mfa_end_shift)
                segments.append({'text': interval.text, 'audio': audio[start_time:end_time]})

        return segments

    @staticmethod
    def save_audio(audio: BytesIO | AudioSegment, name: str) -> None:
        if not isinstance(audio, AudioSegment):
            audio = AudioSegment.from_wav(audio)
        output = BytesIO()
        audio.export(output, format='wav')
        output.seek(0)
        with open(f'{name}.mp3', 'wb') as f:
            f.write(output.read())
