import logging
from io import BytesIO
from itertools import pairwise

import config
import dependencies as container
from google.cloud import storage, texttospeech, texttospeech_v1beta1
from gtts import gTTS
from pydub import AudioSegment
from regex import sub
from TTS.api import TTS


class GTTSProvider:
    '''Simple non-local synthesizer w/o variable parameters.'''

    def __init__(self) -> None:
        self.model = gTTS

    def synthesize(self, text: str, lang: str, *_) -> AudioSegment:
        audio_buffer = BytesIO()
        tts = self.model(text=text, lang=lang)
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return AudioSegment.from_mp3(audio_buffer)


class CoquiTTSProvider:
    '''Local customizable synthesizer with many models support (use multilingual only).

    Require config values: voice_model, source_voice.
    SSML is not officially supported.

    Available models: > tts --list_models
        tts_models/multilingual/multi-dataset/: [xtts_v2, xtts_v1.1, your_tts, bark]

    Available model voices (speakers): > tts --model_name [model] --list_speaker_idxs
    Available model languages (for multilingual): > tts --model_name [model] --list_language_idxs
        tts_models/multilingual/multi-dataset/xtts_v2:
        [en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi]
    '''

    def __init__(self) -> None:
        self.model = TTS(model_name=config.synthesis_model)
        self.voice = config.source_voice

    def synthesize(self, text: str, lang: str, *_) -> AudioSegment:
        audio_buffer = BytesIO()
        self.model.tts_to_file(text=text, file_path=audio_buffer, speaker=self.voice, language=lang)
        audio_buffer.seek(0)
        return AudioSegment.from_wav(audio_buffer)


class GoogleCloudTTSProvider:
    '''Fast cloud speech synthesizer with SSML support.

    Require config values: source_voice, target_voice.

    Available voices: https://cloud.google.com/text-to-speech/docs/voices
    Check src and trg voice types compatibility: https://cloud.google.com/text-to-speech/docs/ssml#select_a_voice

    Save path to your .json credentials to the environment variable 'GOOGLE_APPLICATION_CREDENTIALS'.
    GC.TTS has a quotas for free monthly use.
    '''

    def __init__(self) -> None:
        self.tts = texttospeech_v1beta1 if config.use_ssml == 2 else texttospeech
        self.client_short = self.tts.TextToSpeechClient()
        self.client_long = self.tts.TextToSpeechLongAudioSynthesizeClient()
        self.storage_client = storage.Client()

    def synthesize(self, text: str, lang: str, *, with_timestamps: bool, ssml: bool | None) -> AudioSegment:
        if ssml is True or ssml is None and config.use_ssml:
            synthesis_input = self.tts.SynthesisInput(ssml=text)
            voice_name = config.source_voice
        else:
            synthesis_input = self.tts.SynthesisInput(text=text)
            voice_name = config.source_voice if lang == config.source_lang else config.target_voice
        audio_config = self.tts.AudioConfig(audio_encoding=self.tts.AudioEncoding.LINEAR16)
        lang = voice_name[:5]
        voice = self.tts.VoiceSelectionParams(language_code=lang, name=voice_name)
        if with_timestamps:
            return self._synthesize_with_timestamps(synthesis_input, voice, audio_config)
        if len(text.encode('utf-8')) > 4990:
            logging.debug('GC TTS: Long audio synthesis')
            return self._synthesize_long(synthesis_input, voice, audio_config)
        return self._synthesize_short(synthesis_input, voice, audio_config)

    def _synthesize_with_timestamps(self, synthesis_input, voice, audio_config) -> AudioSegment:
        '''Synthesis with additional timestamps (synthesis_input is ssml with <mark>s).'''
        response = self.client_short.synthesize_speech(
            request=self.tts.SynthesizeSpeechRequest(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
                enable_time_pointing=[self.tts.SynthesizeSpeechRequest.TimepointType.SSML_MARK],
            )
        )
        audio_buffer = BytesIO()
        audio_buffer.write(response.audio_content)
        audio_buffer.seek(0)
        timepoints = [tp.time_seconds * 1000 for tp in response.timepoints]
        return AudioSegment.from_wav(audio_buffer), timepoints

    def _synthesize_short(self, synthesis_input, voice, audio_config) -> AudioSegment:
        '''Simple synthesis with GC TTS.'''
        response = self.client_short.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        audio_buffer = BytesIO()
        audio_buffer.write(response.audio_content)
        audio_buffer.seek(0)
        return AudioSegment.from_wav(audio_buffer)

    def _synthesize_long(self, synthesis_input, voice, audio_config) -> AudioSegment:
        '''"Long" synthesis (segment >5000b) via GC Storage.'''
        bucket = self.storage_client.bucket('sync_book')
        blob = bucket.blob('audio_output.wav')
        if blob.exists():
            blob.delete()

        parent = f'projects/{config.google_cloud_project_id}/locations/{config.google_cloud_project_location}'
        output_gcs_uri = 'gs://sync_book/audio_output.wav'
        request = texttospeech.SynthesizeLongAudioRequest(
            input=synthesis_input, voice=voice, audio_config=audio_config, parent=parent, output_gcs_uri=output_gcs_uri
        )
        operation = self.client_long.synthesize_long_audio(request=request)
        result = operation.result(timeout=300)
        logging.debug(result)
        audio_buffer = BytesIO()
        blob.download_to_file(audio_buffer)
        audio_buffer.seek(0)
        return AudioSegment.from_wav(audio_buffer)


class SpeechSynthesizer:
    def __init__(self) -> None:
        if not set(config.output_types) & {'audio', 'video'}:
            return
        match config.synthesis_provider:
            case 'CoquiTTS':
                self.model = CoquiTTSProvider()
            case 'gTTS':
                self.model = GTTSProvider()
            case 'GoogleCloud':
                self.model = GoogleCloudTTSProvider()
            case value:
                raise ValueError(f'Unknown synthesis_provider value ({value}).')

    def synthesize(self, text: str, lang: str = '', *, with_timestamps=False, ssml=None) -> AudioSegment:
        '''General delegating method.'''
        if text and not config.use_ssml:
            text = sub(r'^\P{L}+', '', text)
            text = sub(r'\P{L}+$', lambda m: m.group(0)[0], text)
        if not text:
            return False
        return self.model.synthesize(text, lang, with_timestamps=with_timestamps, ssml=ssml)

    @staticmethod
    def silent(duration: int = 0) -> AudioSegment:
        return AudioSegment.silent(duration=duration)

    @staticmethod
    def adjust_audio_speed(audio: AudioSegment, speed: float) -> AudioSegment:
        '''Manually change segment speed.'''
        adjusted_audio = BytesIO()
        audio.export(adjusted_audio, format='wav', parameters=['-filter:a', f'atempo={speed}'])
        adjusted_audio.seek(0)
        audio = AudioSegment.from_wav(adjusted_audio)
        return audio

    def synthesize_sentence(
        self, sentence: 'Sentence'
    ) -> tuple[AudioSegment, ...] | tuple[tuple[AudioSegment, list[float]], ...]:
        '''Return synthesized sentences (src and trg) with possible timestamps (if ssml=2).'''
        if not config.use_ssml:
            sent_speed = config.sentence_pronunciation_speed
            src_audio = self.adjust_audio_speed(self.synthesize(sentence.src_text, config.source_lang), sent_speed)
            trg_audio = self.adjust_audio_speed(self.synthesize(sentence.trg_text, config.target_lang), sent_speed)
            return src_audio, trg_audio

        with_timestamps = config.use_ssml == 2
        src_ssml = container.templates['source_sentence'].render(sentence.get_sentence_ssml_config(True))
        trg_ssml = container.templates['target_sentence'].render(sentence.get_sentence_ssml_config(False))
        src_audio = self.synthesize(f'<speak>{src_ssml}</speak>', with_timestamps=with_timestamps)
        trg_audio = self.synthesize(f'<speak>{trg_ssml}</speak>', with_timestamps=with_timestamps)
        return src_audio, trg_audio

    def synthesize_vocabulary(self, sentence: 'Sentence') -> None:
        '''Add separated vocabulary audio (src, trg) to every pair in sentence.result.'''

        def collect_vocabulary_audio(tokens: list['UserToken'], sent_audio: AudioSegment, lang: str) -> AudioSegment:
            '''Compose vocabulary audio from prerecognized segments from sentence audio if possible.'''
            if not config.reuse_synthesized or any(token.audio is None for token in tokens):
                audio = self.synthesize(' '.join(s.text for s in tokens), lang, ssml=False)
                result_audio = self.adjust_audio_speed(audio, config.vocabulary_pronunciation_speed)
                if len(tokens) == 1 and not tokens[0].audio:
                    tokens[0].audio = len(result_audio)
            else:
                result_audio = sent_audio[tokens[0].audio.start : tokens[-1].audio.stop]
                if (speed := config.vocabulary_pronunciation_speed / config.sentence_pronunciation_speed) < 0.5:
                    result_audio = self.adjust_audio_speed(result_audio, 1 / config.sentence_pronunciation_speed)
                    result_audio = self.adjust_audio_speed(result_audio, config.vocabulary_pronunciation_speed)
                else:
                    result_audio = self.adjust_audio_speed(result_audio, speed)

            return result_audio

        if config.use_ssml == 2 and not config.reuse_synthesized:
            # full result at once
            ssml_config = sentence.get_vocabulary_ssml_config()
            voc_ssml = container.templates['vocabulary'].render(ssml_config)
            voc_audio, voc_ts = self.synthesize(f'<speak>{voc_ssml}</speak>', with_timestamps=True)
            prev = None
            for i, (start, stop) in enumerate(pairwise(voc_ts + [len(voc_audio)])):
                if i % 2:
                    sentence.result[i // 2] = (*sentence.result[i // 2], voc_audio[prev], voc_audio[start:stop])
                else:
                    prev = slice(start, stop)
        elif config.use_ssml:
            for i, (src_voc_tokens, trg_voc_tokens) in enumerate(sentence.result[:]):
                audio = []
                for tokens, main_audio, lang, voice in (
                    (src_voc_tokens, sentence.src_audio, config.source_lang, ''),
                    (trg_voc_tokens, sentence.trg_audio, config.target_lang, config.target_voice),
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
                    audio.append(self.synthesize(f'<speak>{voc_ssml}</speak>'))
                sentence.result[i] = (*sentence.result[i], *audio)
        else:
            for i, (src_voc_tokens, trg_voc_tokens) in enumerate(sentence.result[:]):
                src_voc_audio = collect_vocabulary_audio(src_voc_tokens, sentence.src_audio, config.source_lang)
                trg_voc_audio = collect_vocabulary_audio(trg_voc_tokens, sentence.trg_audio, config.target_lang)
                sentence.result[i] = (*sentence.result[i], src_voc_audio, trg_voc_audio)

    def compose_output_audio(self, sentence: 'Sentence') -> AudioSegment:
        '''Compose all audio parts of the sentence with given breaks.'''
        s_break = self.silent(config.break_between_sentences_ms)
        vo_break = self.silent(config.break_between_vocabulary_ms)
        vi_break = self.silent(config.break_in_vocabulary_ms)

        audio = sentence.src_audio + s_break

        if sentence.result:
            result_audio = sentence.result[0][2] + vi_break + sentence.result[0][3]
            for _, _, src_voc_audio, trg_voc_audio in sentence.result[1:]:
                result_audio += vo_break + src_voc_audio + vi_break + trg_voc_audio
            audio += result_audio + s_break

        if sentence.show_translation:
            audio += sentence.trg_audio + s_break
            if config.repeat_original_sentence_after_translated:
                audio += sentence.src_audio + s_break

        return audio

    @staticmethod
    def save_audio(audio: AudioSegment, name: str) -> None:
        audio.export(config.root_dir / f'{name}.wav', format='wav')
