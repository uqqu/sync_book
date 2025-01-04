import warnings
from itertools import pairwise
from os import cpu_count, getenv
from pathlib import Path

import regex as re
from nltk.corpus import stopwords

warnings.filterwarnings('ignore', category=FutureWarning)

# Personal user vocabulary
input_storage_filename = 'user_storage'
output_storage_filename = 'user_storage'
save_results = True

# Main
input_type = 'raw'  # raw, draft  # TODO audio, video, fix_structures
output_types = {'draft', 'text', 'audio', 'video'}  # draft, text, audio, video  # TODO csv, csv_total

translation_provider = 'Argos'  # GoogleCloud, Argos  # TODO DeepL, Yandex
save_translation_to_file = True
use_translation_file = 0  # 0 – False, 1 – raw (align with the literary translation), 2 – presaved
untranslatable_entities = {'PERSON', 'PER', 'DATE'}  # PRODUCT, TIME?

lemma_intervals = tuple(b - a for a, b in pairwise(int(1.05**i) for i in range(200, 2 << 10)))
entity_intervals = tuple(b - a for a, b in pairwise(int(1.1**i) for i in range(200, 2 << 10)))

ignore_warnings = False  # don’t stop if config warnings are detected

# Langs
source_lang = 'en'
target_lang = 'ru'
source_full_lang = 'english'
target_full_lang = 'russian'

# spaCy models
source_model = 'en_core_web_lg'
target_model = 'ru_core_news_lg'  # _lg has 1% better lemmatization than _sm ¯\_(ツ)_/¯

# Token processing
embedding_preprocessing_centering = False
embedding_aggregator = 'attention'  # Averaging, MaxPooling, MinPooling, Attention
min_align_weight = 0.66


# Translated sentence behavior
min_part_of_new_words_to_add = 4  # 1/x from len of original tokens; 0 – never add (prioritized)
min_count_of_new_words_to_add = 2  # 0 – always add
repeat_original_sentence_after_translated = False

# Speech synthesis
synthesis_provider = 'GoogleCloud'  # gTTS, CoquiTTS, GoogleCloud
synthesis_model = 'tts_models/multilingual/multi-dataset/xtts_v2'  # for CoquiTTS only
source_voice = 'en-US-Wavenet-C'  # en-US-Wavenet-C, Tammie Ema  # for GC and CoquiTTS
target_voice = 'ru-RU-Wavenet-C'  # for GC only

sentence_pronunciation_speed = 1
vocabulary_pronunciation_speed = 0.8

break_between_sentences_ms = 400
break_between_vocabulary_ms = 200
break_in_vocabulary_ms = 100

# # for the "long" synthesis (>5000b for segment) with GoogleCloud
google_cloud_project_id = getenv('GCP_ID')
google_cloud_project_location = getenv('GCP_location')

# Audio alignment. To reuse words from a synthesized sentence in a vocabulary or visual alignment on the video
word_break_ms = 100  # between the words in vocabulary (for multiword entities)
crossfade_ms = 20  # on the borders of each word in vocabulary; don't set longer than word_break!
final_silence_of_sentences_ms = 200  # crop final silence for the last word of the sentences
start_shift_ms = 2  # to capture a larger (or smaller) segments
end_shift_ms = 3
use_mfa = False  # the alternative is ssml=2. Don't use them together
mfa_dir = getenv('mfa_path')  # …/MFA/pretrained_models/
mfa_num_jobs = int((cpu_count() or 1) // 1.2)  # you can set your own value as an integer

use_ssml = 2  # currently only works with GC
# 0 – False; 1 – standard; 2 – with timestamps (to reuse pronunciation and more correct video captions)
reuse_synthesized = False  # works correctly only when use_mfa=True xor use_ssml=2

# ssml=1 with mfa=False and reuse=False – is the only case when synthesis is performed for the whole text at once

ssml_vocabulary_pitch = '+10%'  # '' | '{sign}{int}%' | '{int}st' semitones | 'x-low'/'low'/'medium'/'high'/'x-high'
ssml_vocabulary_volume = '-5dB'  # '' | '{sign}{int}dB' | 'silent'/'x-soft'/'soft'/'medium'/'loud'/'x-loud'/'default'
# works only with reuse_synthesized=False and use_ssml in {1, 2}
# if you set the vocabulary speed to 1 and both pitch and volume to '', the <prosody> tag in ssml for vocabulary
# …will be ommited; same for sentence speed only in ssml for sentences. Can be used to limit economy

# Video
caption_font = 'C:/Windows/Fonts/arial.ttf'
video_width = 1800
video_height = 1200
font_size = 36
line_height = 30
bottom_margin = 50
margin_between_original_and_translation = 50



# _Inner
word_pattern = re.compile(r"\p{L}[\p{L}\p{Pd}'ʼ]*\p{L}|\p{L}")
stop_words = set(stopwords.words(source_full_lang)) | set(stopwords.words(target_full_lang))
root_dir = Path(__file__).resolve().parent.parent
