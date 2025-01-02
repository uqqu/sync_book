import subprocess
from pathlib import Path

import config
from tgt.io import read_textgrid


def prepare_alignment(sentences: list['Sentence']) -> None:
    '''Align provided audio buffer with the given text using MFA.'''
    src_data = [(s.src_text, s.src_audio) for s in sentences]
    trg_data = [(s.trg_text, s.trg_audio) for s in sentences]
    for data, lang in ((src_data, config.source_full_lang), (trg_data, config.target_full_lang)):
        temp = config.root_dir / 'temp' / lang
        temp.mkdir(exist_ok=True)
        for i, (text, audio_buffer) in enumerate(data):
            audio_buffer.export(temp / f'temp_{i}.wav', format='wav')
            with open(temp / f'temp_{i}.txt', 'w', encoding='utf-8') as f:
                f.write(text)
        dict_path = Path(config.mfa_dir) / 'dictionary' / f'{lang}_mfa.dict'
        model_path = Path(config.mfa_dir) / 'acoustic' / f'{lang}_mfa.zip'
        command = [
            *('mfa', 'align', '--clean', '--single_speaker', '--fine_tune'),
            *('--num_jobs', str(config.mfa_num_jobs), str(temp), dict_path, model_path, str(temp)),
        ]
        subprocess.run(command, check=True)

def set_alignment(sentence: 'Sentence', idx: int) -> None:
    '''Recognize and set word fragments to token attributes for future addition to audio output.'''

    def _split_audio_by_alignment(max_len: float, lang: str) -> list[dict]:
        '''Split the given audio buffer into segments based on alignment data from a TextGrid file.'''
        segments = []
        textgrid_path = config.root_dir / 'temp' / lang / f'temp_{idx}.TextGrid'
        word_tier = read_textgrid(textgrid_path).get_tier_by_name('words')
        for interval in word_tier.intervals:
            if interval.text.strip():
                start_time = max(0, interval.start_time * 1000 - config.start_shift_ms)
                end_time = min(max_len, interval.end_time * 1000 + config.end_shift_ms)
                segments.append({'text': interval.text, 'audio': slice(start_time, end_time)})

        return segments

    src_segments = _split_audio_by_alignment(len(sentence.src_audio), config.source_full_lang)
    trg_segments = _split_audio_by_alignment(len(sentence.trg_audio), config.target_full_lang)
    for segments, tokens in ((src_segments, sentence.src_tokens), (trg_segments, sentence.trg_tokens)):
        s, t = 0, 0
        while s < len(segments) and t < len(tokens):
            if segments[s]['text'] in tokens[t].text.lower():
                if tokens[t].audio:
                    tokens[t].audio = slice(tokens[t].audio.start, segments[s]['audio'].stop)
                else:
                    tokens[t].audio = segments[s]['audio']
                s += 1
            else:
                t += 1
