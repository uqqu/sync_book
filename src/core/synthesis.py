import logging
from functools import wraps
from io import BytesIO

import regex as re
from google.cloud import storage, texttospeech
from gtts import gTTS
from pydub import AudioSegment
from TTS.api import TTS

import config


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

    def __init__(self) -> None:
        self.model = gTTS

    @adjust_audio_speed
    def synthesize(self, text: str, lang: str) -> AudioSegment:
        audio_buffer = BytesIO()
        tts = self.model(text=text, lang=lang)
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return AudioSegment.from_mp3(audio_buffer)


class CoquiTTSProvider:
    '''Local customizable synthesizer with many models support (use multilingual only).

    Require config values: voice_model, voice_src. Speed optionally (outer regulation).
    SSML is not officially supported.

    Available models: > tts --list_models
        tts_models/multilingual/multi-dataset/: [xtts_v2, xtts_v1.1, your_tts, bark]

    Available model voices (speakers): > tts --model_name [model] --list_speaker_idxs
    Available model languages (for multilingual): > tts --model_name [model] --list_language_idxs
        tts_models/multilingual/multi-dataset/xtts_v2:
        [en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi]
    '''

    def __init__(self) -> None:
        self.model = TTS(model_name=config.synth_model)
        self.voice = config.voice_src

    @adjust_audio_speed
    def synthesize(self, text: str, lang: str) -> AudioSegment:
        audio_buffer = BytesIO()
        self.model.tts_to_file(text=text, file_path=audio_buffer, speaker=self.voice, language=lang)
        audio_buffer.seek(0)
        return AudioSegment.from_wav(audio_buffer)


class GoogleCloudTTSProvider:
    '''Fast cloud speech synthesizer with SSML support.

    Require config values: voice_src, voice_trg. Speed optionally (inner regulation).

    Available voices: https://cloud.google.com/text-to-speech/docs/voices
    Check src and trg voice types compatibility: https://cloud.google.com/text-to-speech/docs/ssml#select_a_voice

    Save path to your .json credentials to the environment variable 'GOOGLE_APPLICATION_CREDENTIALS'.
    GC.TTS has a quotas for free monthly use.
    '''

    def __init__(self) -> None:
        self.client_short = texttospeech.TextToSpeechClient()
        self.client_long = texttospeech.TextToSpeechLongAudioSynthesizeClient()
        self.storage_client = storage.Client()

    def synthesize(self, text: str, lang: str, speed: float) -> AudioSegment:
        if config.use_ssml and not config.use_mfa:
            input_text = texttospeech.SynthesisInput(ssml=text)
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
            voice_name = config.voice_src
        else:
            input_text = texttospeech.SynthesisInput(text=text)
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16, speaking_rate=speed
            )
            voice_name = config.voice_src if lang == config.source_lang else config.voice_trg
        lang = voice_name[:5]
        voice = texttospeech.VoiceSelectionParams(language_code=lang, name=voice_name)
        if len(text) > 4990:
            logging.debug('GC TTS: Long audio synthesis')
            return self._synthesize_long(input_text, voice, audio_config)
        return self._synthesize_short(input_text, voice, audio_config)

    def _synthesize_short(self, input, voice, audio_config) -> AudioSegment:
        response = self.client_short.synthesize_speech(input=input, voice=voice, audio_config=audio_config)
        audio_buffer = BytesIO()
        audio_buffer.write(response.audio_content)
        audio_buffer.seek(0)
        return AudioSegment.from_wav(audio_buffer)

    def _synthesize_long(self, input, voice, audio_config) -> AudioSegment:
        bucket = self.storage_client.bucket('sync_book')
        blob = bucket.blob('audio_output.wav')
        if blob.exists():
            blob.delete()

        parent = f'projects/{config.google_cloud_project_id}/locations/{config.google_cloud_project_location}'
        output_gcs_uri = 'gs://sync_book/audio_output.wav'
        request = texttospeech.SynthesizeLongAudioRequest(
            input=input, voice=voice, audio_config=audio_config, parent=parent, output_gcs_uri=output_gcs_uri
        )
        operation = self.client_long.synthesize_long_audio(request=request)
        result = operation.result(timeout=300)
        logging.debug(
            'Finished processing, check your GCS bucket to find your audio file!',
            'Printing what should be an empty result: ',
            result,
        )
        audio_buffer = BytesIO()
        blob.download_to_file(audio_buffer)
        audio_buffer.seek(0)
        return AudioSegment.from_wav(audio_buffer)


class SpeechSynthesizer:
    def __init__(self) -> None:
        if not config.speech_synth:
            return
        match config.synth_provider:
            case 'CoquiTTS':
                self.model = CoquiTTSProvider()
            case 'gTTS':
                self.model = GTTSProvider()
            case 'GoogleCloud':
                self.model = GoogleCloudTTSProvider()
            case _:
                raise ValueError(f'Unknown synth_provider value ({config.synth_provider}).')

    def synthesize(self, text: str, lang: str, speed: float = 1.0) -> AudioSegment:
        if text and not config.use_ssml:
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
                    audio = self.synthesize(value, config.source_lang, speed)
                case 1:
                    audio = self.synthesize(value, config.target_lang, speed)
                case 2:
                    audio = self.synthesize_by_parts(
                        [(i, v) for val in value for i, v in enumerate(val)], config.vocabulary_pronunciation_speed
                    )
                case _:
                    continue
            if audio:
                audio_buffer += audio
                audio = None
        return audio_buffer

    @staticmethod
    def save_audio(audio: AudioSegment, name: str) -> None:
        audio.export(config._root_dir / f'{name}.wav', format='wav')
