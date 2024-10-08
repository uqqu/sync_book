import io
import logging
import regex as re

from gtts import gTTS


class SpeechSynthesizer:
    def __init__(self, src_lang: str, trg_lang: str, lang_delimeter: str):
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.lang_delimeter = lang_delimeter

    def save_tts(self, text):
        final_audio = io.BytesIO()
        lang_flag = False

        for segment_text in ''.join(text).split(self.lang_delimeter):
            lang_flag = not lang_flag
            segment_text = re.sub(r'^\P{L}+|\P{L}+$', '', segment_text)
            if not segment_text:
                continue
            try:
                tts = gTTS(text=segment_text, lang=(self.src_lang if lang_flag else self.trg_lang))
            except Exception as e:
                logging.error(f'Synthesis error with segment "{segment_text}": {e}')
                continue
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)

            audio_bytes.seek(0)
            if not final_audio.tell():
                final_audio.write(audio_bytes.read())
            else:
                audio_bytes.read(10)
                final_audio.write(audio_bytes.read())

        final_audio.seek(0)
        with open('multilingual_output.mp3', 'wb') as f:
            f.write(final_audio.read())
