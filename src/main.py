import json
import logging
import sys
from dataclasses import asdict
from os import cpu_count

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

        for attr in ('embedding', 'position'):
            if not Token.has_extension(attr):
                Token.set_extension(attr, default=None)

        if config.use_mfa or config.save_translation_to_file or config.use_translation_file == 2:
            (config.root_dir / 'temp').mkdir(exist_ok=True)

    def process(self) -> None:
        self.prepare_sentences()

        if 'draft' in config.output_types:
            self.dump_draft()

        for sent in self.sentences:
            sent.update_positions()

        if config.save_results:
            container.structures.save_structures()

        if 'text' in config.output_types:
            print(self.get_output_text())

        if 'audio' in config.output_types or 'video' in config.output_types:
            audio = self.get_output_audio()
            container.synthesizer.save_audio(audio, 'multilingual_output')
            logging.info('Final audio was generated and saved to root folder as "multilingual_output.mp3".')

        if 'video' in config.output_types:
            for sentence in self.sentences:
                container.video.append_sentence_to_clips(sentence)
            video = container.video.compose_clips()
            container.video.save_video(video, 'video_output')
            logging.info('Final video was generated and saved to root folder as "video_output.mp4".')

    def prepare_sentences(self) -> None:
        '''Prepare sentences on a given input type.'''
        match config.input_type:
            case 'raw':
                with open(config.root_dir / 'input_text.txt', 'r', encoding='utf-8') as textfile:
                    input_text = textfile.read()
                logging.info('Start translating the sentences. It may take some time.')
                sentences = container.translator.process_sentences(
                    container.text_preprocessor.get_sentences(input_text)
                )
                if config.save_translation_to_file and config.use_translation_file != 2:
                    container.translator.save_translation_to_file()

                for src_text in sentences:
                    trg_text = container.translator.get_translated_sentence(src_text)
                    src_tokens, trg_tokens = container.text_preprocessor.get_tokens_with_embeddings(src_text, trg_text)
                    tokens = TokenProcessing(src_tokens, trg_tokens).process()
                    self.sentences.append(Sentence(src_text, trg_text, tokens))
                    container.structures.sentence_counter += 1

            case 'draft':
                transform_tokens = lambda tokens: [
                    UserToken(**{attr: token[attr] for attr in ('text', 'lemma_', 'index', 'position', 'is_punct')})
                    for token in tokens
                ]
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

            case 'audio':  # TODO
                pass

            case 'video':  # TODO
                pass

            case value:
                raise RuntimeError(f'Incorrect input_type value on config.py – "{value}".')

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

    def get_output_text(self) -> str:
        '''Return the processed text in its final form, as it will be used for speech synthesis.'''
        text = []
        for sent in self.sentences:
            stripped = sent.src_text.rstrip()
            tail = sent.src_text[len(stripped) :]
            text.append(stripped)
            if sent.vocabulary:
                voc = (
                    f'{" ".join(s.text for s in src)}–{" ".join(t.text for t in trg)}' for src, trg in sent.vocabulary
                )
                text.append(f'[{", ".join(voc)}]')
            if sent.show_translation:
                text.append(sent.trg_text)
                if config.repeat_original_sentence_after_translated:
                    text.append(stripped)
            if tail:
                text.append(tail)
        return ' '.join(text)

    def get_output_audio(self) -> 'AudioSegment':
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

        return audio


def check_config_errors():
    '''Detect unacceptable combinations of configuration attributes.'''
    c = config
    errors = {
        (c.use_mfa and c.use_ssml == 2, 'Ambiguous behavior: mfa=True; use_ssml=2. Choose only one.'),
        (not 0.5 <= c.sentence_pronunciation_speed <= 2, 'Set sentence_pronunciation_speed in range of 0.5-2.'),
        (not 0.5 <= c.vocabulary_pronunciation_speed <= 2, 'Set vocabulary_pronunciation_speed in range of 0.5-2.'),
        (c.use_mfa and c.crossfade_ms > c.word_break_ms, 'A crossfade can’t be longer than a word_break.'),
        (c.use_mfa and not c.mfa_dir, 'Empty environment variable "mfa_path".'),
        (c.use_ssml and c.synthesis_provider != 'GoogleCloud', 'The selected synthesis provider doesn’t support ssml.'),
    }

    has_errors = False
    for condition, message in errors:
        if condition:
            print(f'[‽] Error: {message}')
            has_errors = True
    if has_errors:
        raise RuntimeError('Config validation failed due to errors!')


def check_config_warnings():
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
            'video' in config.output_types and not (config.root_dir / 'background.jpg').exists(),
            'File "background.jpg" was not found in the root project directory. Video will be generated with '
            'solid gray background.'
        )
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
    app.process()
