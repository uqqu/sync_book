from itertools import pairwise
from os import getenv
from pathlib import Path

import regex as re
from nltk.corpus import stopwords

source_model = 'en_core_web_sm'
target_model = 'ru_core_news_sm'

source_lang = 'en'
target_lang = 'ru'
source_full_lang = 'english'
target_full_lang = 'russian'

embedding_preprocessing_center = False
embedding_aggregator = 'attention'  # Averaging, MaxPooling, MinPooling, Attention
min_align_weight = 0.66

lemma_intervals = tuple(b - a for a, b in pairwise(int(1.05**i) for i in range(190, 600)))
entity_intervals = tuple(b - a for a, b in pairwise(int(1.1**i) for i in range(190, 600)))

stop_words = set(stopwords.words(source_full_lang))
stop_words |= set(stopwords.words(target_full_lang))
untranslatable_entities = {'PERSON', 'DATE'}  # TODO PRODUCT, TIME?
word_pattern = re.compile(r"^\p{L}[\p{L}\p{Pd}'’ʼ]*\p{L}$")

input_storage_filename = 'user_storage'
output_storage_filename = 'user_storage'
save_results = True

use_translation_file = 0  # 0 – False, 1 – raw, 2 – presaved
save_translation_to_file = True
translation_provider = 'Argos'  # GoogleCloud, Argos

# translated sentence behavior
min_new_words_part_to_add = 4  # 1/x from len of original tokens; 0 – never add (prioritized)
min_new_words_count_to_add = 2  # 0 – always add
repeat_original_sentence_after_translated = True

# Speech synthesis config
speech_synth = True
synth_provider = 'gTTS'  # gTTS, CoquiTTS, GoogleCloud
synth_model = 'tts_models/multilingual/multi-dataset/xtts_v2'
voice_src = 'en-US-Wavenet-C'  # Tammie Ema, en-US-Wavenet-C
voice_trg = 'ru-RU-Wavenet-C'
sentence_pronunciation_speed = 0.9
vocabulary_pronunciation_speed = 0.8
use_ssml = False

# for the synthesis long audio (>5000b) with GoogleCloud
google_cloud_project_id = getenv('GCP_ID')
google_cloud_project_location = getenv('GCP_location')

# synthesis with post audio alignment to reuse words for the dictionary from generated sentence
use_mfa = False  # extremely long and jerkily :x
mfa_dir = getenv('mfa_path')  # …/MFA/pretrained_models/
mfa_start_shift_ms = 2
mfa_end_shift_ms = 3


_root_dir = Path(__file__).resolve().parent.parent
