import csv
import json
import logging
import subprocess
import sys
from dataclasses import asdict
from os import cpu_count
from shutil import copy

from spacy.tokens import Token

import config
import core.mfa_aligner as MFAligner
import dependencies as container
from core._structures import UserToken
from core.sentence_processing import Sentence
from core.token_processing import TokenProcessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Main:
    def __init__(self) -> None:
        self.sentences: list[Sentence] = []

        (config.root_dir / 'temp').mkdir(exist_ok=True)
        for attr in ('embedding', 'position'):
            if not Token.has_extension(attr):
                Token.set_extension(attr, default=None)

        try:
            {'text': self.process_input_text, 'draft': self.process_input_draft}[config.input_type]()
        except KeyError:
            raise RuntimeError(f'Incorrect input_type value on config.py – "{config.input_type}".')

    def process_input_text(self) -> None:
        '''Create sentences from input text file.'''
        with open(config.root_dir / 'input_text.txt', 'r', encoding='utf-8') as textfile:
            input_text = textfile.read().strip()
        logging.info('Start translating the sentences. It may take some time.')
        sentences = container.translator.process_sentences(container.text_preprocessor.get_sentences(input_text))
        container.translator.save_translation_to_file()

        for src_text in sentences:
            trg_text = container.translator.get_translated_sentence(src_text)
            src_tokens, trg_tokens = container.text_preprocessor.get_tokens_with_embeddings(src_text, trg_text)
            if not src_tokens or not trg_tokens:
                continue
            tokens = TokenProcessing(src_tokens, trg_tokens).process()
            self.sentences.append(Sentence(src_text, trg_text, tokens))
            container.structures.sentence_counter += 1

    def process_input_draft(self) -> None:
        '''Read sentences from user draft.'''
        attrs = ('text', 'lemma_', 'index', 'position', 'is_punct', 'whitespace')
        transform_tokens = lambda tokens: [UserToken(**{a: token[a] for a in attrs}) for token in tokens]
        with open(config.root_dir / 'draft.json', 'r', encoding='utf-8') as file:
            struct = json.load(file)
        logging.info('Got the sentences from the draft.')
        for sent in struct:
            converted_tokens = [
                transform_tokens(sent['src_tokens']),
                transform_tokens(sent['trg_tokens']),
                [(transform_tokens(s), transform_tokens(t), status) for s, t, status in sent['vocabulary']],
            ]
            self.sentences.append(
                Sentence(sent['src_text'], sent['trg_text'], converted_tokens, sent['show_translation'])
            )

    def get_outputs(self) -> None:
        '''Get all outputs by user config in strict order.'''
        config.output_types |= {'save', 'update'} if config.save_results else {'update'}

        outputs = {
            'csv_b': self.save_output_csv_total,
            'draft': self.dump_draft,
            'update': self.sentence_postprocessing,
            'save': container.structures.save_structures,
            'csv_a': self.save_output_csv_total,
            'csv_c': self.save_output_csv,
            'text': self.save_output_text,
            'audio': self.save_output_audio,
            'subs': self.save_output_subs,
            'video': self.save_output_video,
        }

        for name, func in outputs.items():
            if name in config.output_types:
                func()

    def sentence_postprocessing(self) -> None:
        '''Retry to update positions and postcombine entities.'''
        for sentence in self.sentences:
            sentence.update_positions()
            if config.vocabulary_postcombining and config.postcombining_max_num:
                sentence.vocabulary_postcombining()

    def dump_draft(self) -> None:
        '''Save draft as json file for manual changes and reusing it with config.input_type = "draft".'''

        def pack(tokens: list[UserToken]) -> list[dict[str, str | int | bool]]:
            return [{k: v for k, v in asdict(token).items() if k not in {'audio', 'segments'}} for token in tokens]

        json_output = []
        for sentence in self.sentences:
            json_output.append(
                {
                    'src_text': sentence.src_text,
                    'trg_text': sentence.trg_text,
                    'src_tokens': pack(sentence.src_tokens),
                    'trg_tokens': pack(sentence.trg_tokens),
                    'show_translation': sentence.show_translation,
                    'vocabulary': [
                        (pack(src_voc), pack(trg_voc), status) for src_voc, trg_voc, status in sentence.vocabulary
                    ],
                }
            )
        with open(config.root_dir / 'draft.json', 'w', encoding='utf-8') as file:
            json.dump(json_output, file, ensure_ascii=False, indent=2)
        logging.info(
            'Sentences have been dumped into draft.json. '
            'Now you can make direct changes and pass them as input data.'
        )

    def save_output_csv(self) -> None:
        '''Save .csv with data from this generation sentences.'''
        with open(config.root_dir / 'output_csv.csv', 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            for sent in self.sentences:
                for src, trg in sent.vocabulary:
                    writer.writerow([' '.join(x.text for x in src), ' '.join(x.text for x in trg)])

    def save_output_csv_total(self) -> None:
        '''Save .csv with pairs from full structures.'''
        with open(config.root_dir / 'output_csv_total.csv', 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            for src, trg in container.structures.get_pairs():
                writer.writerow([src, trg])

    def save_output_text(self) -> None:
        '''Return the processed text in its final form, as it will be used for speech synthesis.'''
        text = []
        for sent in self.sentences:
            stripped = sent.src_text.rstrip()
            tail = sent.src_text[len(stripped) :]
            text.append(stripped)
            if sent.vocabulary:
                joined = lambda x: ''.join(f'{t.text} ' if t.whitespace else t.text for t in x[:-1]) + x[-1].text
                text.append(f'[{"; ".join(f"{joined(src)}–{joined(trg)}" for src, trg in sent.vocabulary)}]')
            if sent.show_translation:
                text.append(sent.trg_text)
                if config.repeat_original_sentence_after_translated:
                    text.append(stripped)
            if tail:
                text.append(tail)
        text = ' '.join(text)
        logging.info(text)
        with open(config.root_dir / 'output_text.txt', 'w', encoding='utf-8') as txt:
            txt.write(text)

    def save_output_audio(self) -> None:
        '''Synthesize and save final form of the text, according to the specified parameters.'''
        logging.info('Start audio synthesis...')
        for sentence in self.sentences:
            sentence.synthesize_sentence()
        if config.use_mfa:
            MFAligner.prepare_alignment(self.sentences)

        audio = container.synthesizer.silent()
        for i, sentence in enumerate(self.sentences):
            if config.use_mfa:
                MFAligner.set_alignment(sentence, i)
            sentence.synthesize_vocabulary()
            audio += container.synthesizer.compose_output_audio(sentence)

        container.synthesizer.save_audio(audio)
        logging.info('Final audio was generated and saved to root folder as "output_audio.wav".')

    def save_output_subs(self) -> None:
        '''Save subtitles .ass file with highlight formatting for video generation with ffmpeg or just to have.'''
        with open(config.root_dir / 'output_subtitles.ass', 'w', encoding='utf-8') as subs:
            subs.write(container.video.generate_subtitles(self.sentences))

    def save_output_video(self) -> None:
        '''Save video with ffmpeg hardsubbing or moviepy manual text placement.'''
        if config.manual_subs:
            for sentence in self.sentences:
                container.video.append_sentence_to_clips(sentence)
            video = container.video.compose_clips()
            container.video.save_video(video)
            logging.info('Final video was generated and saved to root folder as "output_video.mp4".')
            return

        if 'subs' in config.output_types or 'audio' in config.presaved:
            copy(config.root_dir / 'output_subtitles.ass', config.root_dir / 'src' / 'subtitles.ass')
        else:
            with open(config.root_dir / 'src' / 'subtitles.ass', 'w', encoding='utf-8') as subs:
                subs.write(container.video.generate_subtitles(self.sentences))
        command = (
            f'ffmpeg -loop 1 -i "{config.root_dir / "background.jpg"}" '
            f'-i "{config.root_dir / "output_audio.wav"}" '
            f'-vf "subtitles=subtitles.ass" '  # doesn’t support paths, only nearby files
            f'-c:v libx264 -c:a aac -shortest -preset ultrafast -pix_fmt yuv420p -y '
            f'"{config.root_dir / "output_video.mp4"}"'
        )
        subprocess.run(command, check=True)
        (config.root_dir / 'src' / 'subtitles.ass').unlink()


def check_config_errors() -> None:
    '''Detect unacceptable combinations of configuration attributes.'''
    c = config
    errors = {
        (c.use_mfa and c.use_ssml == 2, 'Ambiguous behavior: mfa=True; use_ssml=2. Choose only one.'),
        (not 0.5 <= c.sentence_pronunciation_speed <= 2, 'Set sentence_pronunciation_speed in range of 0.5-2.'),
        (not 0.5 <= c.vocabulary_pronunciation_speed <= 2, 'Set vocabulary_pronunciation_speed in range of 0.5-2.'),
        (c.use_mfa and not c.mfa_dir, 'Empty environment variable "mfa_path".'),
        (c.use_ssml and c.synthesis_provider != 'GoogleCloud', 'The selected synthesis provider doesn’t support ssml.'),
        (
            'video' in c.output_types and not c.manual_subs and not (c.root_dir / 'background.jpg').exists(),
            'File "background.jpg" was not found in the root project directory. With ffmpeg hardsubs it’s necessary.',
        ),
        (
            'audio' in c.presaved
            and 'video' in c.output_types
            and (not (c.root_dir / 'output_audio.wav').exists() or not (c.root_dir / 'output_subtitles.ass').exists()),
            'Audio reusing requires audio and subs files from previous generation in the root project directory.',
        ),
        (
            'audio' in c.presaved and 'video' in c.output_types and c.manual_subs,
            'Audio cannot be reused with manual_subs=True.',
        ),
        (
            'audio' in c.presaved and ('subs' in c.output_types or 'audio' in c.output_types),
            'You should exclude audio and subs from output if you want to use presaved audio.',
        ),
    }

    has_errors = False
    for condition, message in errors:
        if condition:
            print(f'[‽] Error: {message}')
            has_errors = True
    if has_errors:
        raise RuntimeError('Config validation failed due to errors!')


def check_config_warnings() -> None:
    '''Detect of possibly undesirable combinations of configuration attributes.'''
    c = config
    warnings = {
        (
            c.output_types == {'draft'} and c.save_results,
            'You set only draft as output with save_results as True. Perhaps you would like to not save '
            'the result without other outputs?',
        ),
        (
            c.synthesis_provider == 'GoogleCloud' and not c.google_cloud_project_id,
            'Empty environment variable "GCP_ID". Long synthesis (segments >5000b) will end in an error.',
        ),
        (
            c.synthesis_provider == 'GoogleCloud' and not c.google_cloud_project_location,
            'Empty environment variable "GCP_location". Long synthesis (segments >5000b) will end in an error.',
        ),
        (
            not c.reuse_synthesized and c.use_ssml == 2,
            'You specify timestamped ssml synthesis without reusing tokens. It can be really voluminous '
            'for quotas. If you are not sure you need timestamps, abort the run and change the configuration.',
        ),
        (
            c.reuse_synthesized and c.use_ssml != 2 and not c.use_mfa,
            'Reusing words for vocabulary without using mfa or ssml with timestamps (2) will lead to a '
            'rather approximate evaluation of the audio segments, and will significantly affect the final sound.',
        ),
        (
            c.use_mfa and c.mfa_num_jobs < cpu_count() >> 1,
            f'You can specify a higher value on mfa_num_jobs to speed up the process (up to {cpu_count()}).',
        ),
        (
            not c.min_part_of_new_words_to_add,
            'You have set min_part_of_new_words_to_add to 0. Translated sentences will never be displayed.',
        ),
        (
            c.min_part_of_new_words_to_add and not c.min_count_of_new_words_to_add,
            'You have set min_count_of_new_words_to_add to 0. Each sentence will be accompanied by a translation.',
        ),
        (
            not 0.5 <= (c.sentence_pronunciation_speed / c.vocabulary_pronunciation_speed) <= 2,
            'Major difference between sentence and vocabulary pronunciation speed. It can cause sound distortion.',
        ),
        (
            'video' in c.output_types and c.manual_subs and not (c.root_dir / 'background.jpg').exists(),
            'File "background.jpg" was not found in the root project directory. Video will be generated with '
            'solid gray background.',
        ),
    }

    has_warnings = False
    for condition, message in warnings:
        if condition:
            print(f'[!] Warning: {message}')
            has_warnings = True
    if has_warnings and not c.ignore_warnings:
        while (ans := input('Do you want to continue processing? [y/n]: ')) not in {'y', 'n'}:
            continue
        if ans == 'n':
            sys.exit()


if __name__ == '__main__':
    check_config_errors()
    check_config_warnings()
    app = Main()
    app.get_outputs()
