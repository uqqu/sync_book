import logging
from functools import wraps
from io import BytesIO

import regex as re
from google.cloud import texttospeech
from gtts import gTTS
from pydub import AudioSegment
from TTS.api import TTS


def adjust_audio_speed(func):
    '''For the most providers this is the only way to adjust speech rate.'''

    @wraps(func)
    def wrapper(self, text: str, lang: str, speed: float) -> AudioSegment:
        audio: AudioSegment = func(self, text, lang)
        if speed != 1:
            adjusted_audio = BytesIO()
            audio.export(adjusted_audio, format='wav', parameters=['-filter:a', f'atempo={speed}'])
            adjusted_audio.seek(0)
            audio = AudioSegment.from_wav(adjusted_audio)
        return audio

    return wrapper


class GTTSProvider:
    '''Simple non-local synthesizer w/o variable parameters.'''

    def __init__(self, _) -> None:
        self.model = gTTS

    @adjust_audio_speed
    def synthesize(self, text: str, lang: str) -> AudioSegment:
        audio_buffer = BytesIO()
        tts = self.model(text=text, lang=lang)
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return AudioSegment.from_wav(audio_buffer)


class CoquiTTSProvider:
    '''Local customizable synthesizer with many models support (use multilingual only).

    Require speech_config values: model, voice_src. Speed optionally (outer regulation).
    SSML is not officially supported.

    Available models: > tts --list_models
        tts_models/multilingual/multi-dataset/: [xtts_v2, xtts_v1.1, your_tts, bark]

    Available model voices (speakers): > tts --model_name [model] --list_speaker_idxs
    Available model languages (for multilingual): > tts --model_name [model] --list_language_idxs
        tts_models/multilingual/multi-dataset/xtts_v2:
        [en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi]
    '''

    def __init__(self, speech_config: 'SpeechConfig') -> None:
        self.model = TTS(model_name=speech_config.model)
        self.voice = speech_config.voice_src

    @adjust_audio_speed
    def synthesize(self, text: str, lang: str) -> AudioSegment:
        audio_buffer = BytesIO()
        self.model.tts_to_file(text=text, file_path=audio_buffer, speaker=self.voice, language=lang)
        audio_buffer.seek(0)
        return AudioSegment.from_wav(audio_buffer)


class GoogleCloudTTSProvider:
    '''Fast cloud speech synthesizer with SSML support.

    Available voices: https://cloud.google.com/text-to-speech/docs/voices
    Check src and trg voice types compatibility: https://cloud.google.com/text-to-speech/docs/ssml#select_a_voice

    Save path to your .json credentials to the environment variable 'GOOGLE_APPLICATION_CREDENTIALS'.
    GC.TTS has a quotas for free monthly use.

    Require speech_config values by keys: voice_src, voice_trg. Speed optionally (inner regulation).
    '''

    def __init__(self, speech_config: 'SpeechConfig') -> None:
        self.speech_config = speech_config
        self.model = texttospeech.TextToSpeechClient()

    def synthesize(self, text: str, lang: str, speed: float) -> AudioSegment:
        if self.speech_config.ssml:
            input_text = texttospeech.SynthesisInput(ssml=text)
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
            voice_name = self.speech_config.voice_src
        else:
            input_text = texttospeech.SynthesisInput(text=text)
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16, speaking_rate=speed
            )
            voice_name = (
                self.speech_config.voice_src
                if lang == self.speech_config.voice_src[:2]
                else self.speech_config.voice_trg
            )
        lang = voice_name[:5]
        voice = texttospeech.VoiceSelectionParams(language_code=lang)
        response = self.model.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
        audio_buffer = BytesIO()
        audio_buffer.write(response.audio_content)
        audio_buffer.seek(0)
        return AudioSegment.from_wav(audio_buffer)


class SpeechSynthesizer:
    def __init__(self, config: 'Config') -> None:
        if not config.speech_synth:
            return
        self.config = config
        match config.speech_config.provider:
            case 'CoquiTTS':
                provider = CoquiTTSProvider
            case 'gTTS':
                provider = GTTSProvider
            case 'GoogleCloud':
                provider = GoogleCloudTTSProvider
            case _:
                raise ValueError(f'Unknown speech_synth_provider value ({config.speech_config.provider}).')
        self.model = provider(config.speech_config)

    def synthesize(self, text: str, lang: str, speed: float = 1.0) -> AudioSegment:
        if text and not self.config.speech_config.ssml:
            text = re.sub(r'^\P{L}+|[\P{L}?!\.]+$', '', text)
        if not text:
            return False
        return self.model.synthesize(text, lang, speed)

    @staticmethod
    def silent(duration: int) -> AudioSegment:
        return AudioSegment.silent(duration=duration)

    def synthesize_by_parts(self, parts: list[tuple[int, str | list[tuple[str, str]]]], speed: float) -> AudioSegment:
        audio_buffer = self.silent(0)
        for flag, value in parts:
            match flag:
                case 0:
                    audio = self.synthesize(value, self.config.source_lang, speed)
                case 1:
                    audio = self.synthesize(value, self.config.target_lang, speed)
                case 2:
                    audio = self.synthesize_by_parts(
                        [(i, v) for val in value for i, v in enumerate(val)], self.config.speech_config.vocabulary_speed
                    )
                case _:
                    continue
            if audio:
                audio_buffer += audio
                audio = None
        return audio_buffer

    @staticmethod
    def save_audio(audio: AudioSegment, name: str) -> None:
        audio.export(f'{name}.wav', format='wav')
