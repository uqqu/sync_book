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
        self.lang_delimiter = ''  # '\u200B'

        self.min_align_weight = 0.6
        self.lemma_intervals = tuple(b - a for a, b in pairwise(int(1.05**i) for i in range(190, 600)))
        self.entity_intervals = tuple(b - a for a, b in pairwise(int(1.1**i) for i in range(190, 600)))

        self.stop_words = set(stopwords.words(self.source_full_lang))
        self.stop_words |= set(stopwords.words(self.target_full_lang))
        self.untranslatable_entities = {'PERSON', 'DATE'}  # TODO PRODUCT, TIME?
        self.word_pattern = re.compile(r"^\p{L}[\p{L}\p{Pd}'’ʼ]*\p{L}$")

        self.speech_synth = True

        # synthesis with post audio alignment to reuse words for the dictionary from generated sentence
        self.mfa_use = False  # extremely long and jerkily :x
        self.mfa_dir = os.getenv('mfa_path')  # …/MFA/pretrained_models/
        self.mfa_start_shift = 10  # ms
        self.mfa_end_shift = 30  # ms
