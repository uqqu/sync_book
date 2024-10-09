from itertools import pairwise

import regex as re
from nltk.corpus import stopwords


class Config:
    def __init__(self):
        self.source_model = 'en_core_web_sm'
        self.target_model = 'ru_core_news_sm'

        self.source_lang = 'en'
        self.target_lang = 'ru'
        self.lang_delimeter = '\u200B'
        self.stop_words = set(stopwords.words('english')) | set(stopwords.words('russian'))

        self.min_align_weight = 0.6
        self.manual_min_align_weight = 0.7

        self.lemma_intervals = tuple(b - a for a, b in pairwise(int(1.05**i) for i in range(190, 600)))
        self.entity_intervals = tuple(b - a for a, b in pairwise(int(1.1**i) for i in range(190, 600)))

        self.untranslatable_entities = {'PERSON', 'DATE'}  # PRODUCT, TIME?
        self.word_pattern = re.compile(r"^\p{L}[\p{L}\p{Pd}'’ʼ]*\p{L}$")

        self.speech_synth = False
