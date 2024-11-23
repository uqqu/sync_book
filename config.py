import os
from itertools import pairwise

import regex as re
from nltk.corpus import stopwords

source_model = 'en_core_web_sm'
target_model = 'ru_core_news_sm'

source_lang = 'en'
target_lang = 'ru'
source_full_lang = 'english'
target_full_lang = 'russian'

embedding_preprocessing_center = False
embedding_aggregator = 'averaging'  # Averaging, MaxPooling, MinPooling, Attention
min_align_weight = 0.7

lemma_intervals = tuple(b - a for a, b in pairwise(int(1.05**i) for i in range(190, 600)))
entity_intervals = tuple(b - a for a, b in pairwise(int(1.1**i) for i in range(190, 600)))

stop_words = set(stopwords.words(source_full_lang))
stop_words |= set(stopwords.words(target_full_lang))
untranslatable_entities = {'PERSON', 'DATE'}  # TODO PRODUCT, TIME?
word_pattern = re.compile(r"^\p{L}[\p{L}\p{Pd}'’ʼ]*\p{L}$")

input_storage_filename = 'lemmas'
output_storage_filename = 'lemmas'
save_results = False

use_translation_file = 0  # 0 – False, 1 – raw, 2 – presaved
save_translation_to_file = True
translation_provider = 'Argos'  # GoogleCloud, Argos


# Speech synthesis config
speech_synth = False
synth_provider = 'GoogleCloud'  # gTTS, CoquiTTS, OpenTTS, GoogleCloud
synth_model = 'tts_models/multilingual/multi-dataset/xtts_v2'
voice_src = 'en-US-Wavenet-C'  # Tammie Ema, en-US-Wavenet-C
voice_trg = 'ru-RU-Wavenet-C'
sentence_pronunciation_speed = 0.9
vocabulary_pronunciation_speed = 0.7
use_ssml = False

# for the synthesis long audio (>5000b) with GoogleCloud
google_cloud_project_id = os.getenv('GCP_ID')
google_cloud_project_location = os.getenv('GCP_location')

# synthesis with post audio alignment to reuse words for the dictionary from generated sentence
use_mfa = False  # extremely long and jerkily :x
mfa_dir = os.getenv('mfa_path')  # …/MFA/pretrained_models/
mfa_start_shift_ms = 5
mfa_end_shift_ms = 10
