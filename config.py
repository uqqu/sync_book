import os
from itertools import pairwise

import regex as re
from nltk.corpus import stopwords


class Config:
    def __init__(self):
        self.source_model = 'en_core_web_sm'
        self.target_model = 'ru_core_news_sm'

        self.source_lang = 'en'
        self.target_lang = 'ru'
        self.source_full_lang = 'english'
        self.target_full_lang = 'russian'

        self.min_align_weight = 0.7
        self.lemma_intervals = tuple(b - a for a, b in pairwise(int(1.05**i) for i in range(190, 600)))
        self.entity_intervals = tuple(b - a for a, b in pairwise(int(1.1**i) for i in range(190, 600)))

        self.stop_words = set(stopwords.words(self.source_full_lang))
        self.stop_words |= set(stopwords.words(self.target_full_lang))
        self.untranslatable_entities = {'PERSON', 'DATE'}  # TODO PRODUCT, TIME?
        self.word_pattern = re.compile(r"^\p{L}[\p{L}\p{Pd}'’ʼ]*\p{L}$")

        self.input_storage_filename = 'lemmas'
        self.output_storage_filename = 'lemmas'
        self.save_results = False
        self.use_translation_file = False
        self.translation_provider = 'GoogleCloud'  # GoogleCloud, Argos

        self.speech_synth = True
        self.speech_config = SpeechConfig()


class SpeechConfig:
    def __init__(self):
        self.provider = 'CoquiTTS'  # gTTS, CoquiTTS, OpenTTS, GoogleCloud
        self.model = 'tts_models/multilingual/multi-dataset/xtts_v2'
        self.voice_src = 'Tammie Ema'  # Tammie Ema, en-US-Wavenet-C
        self.voice_trg = 'ru-RU-Wavenet-C'
        self.ssml = False
        self.sentence_speed = 0.9
        self.vocabulary_speed = 0.7

        # synthesis with post audio alignment to reuse words for the dictionary from generated sentence
        self.mfa_use = False  # extremely long and jerkily :x
        self.mfa_dir = os.getenv('mfa_path')  # …/MFA/pretrained_models/
        self.mfa_start_shift = 10  # ms
        self.mfa_end_shift = 30  # ms
