import warnings
from itertools import pairwise
from os import cpu_count, getenv
from pathlib import Path

import regex as re
from nltk.corpus import stopwords

warnings.filterwarnings('ignore', category=FutureWarning)

# Personal user vocabulary
storage_filename = 'user_storage'
save_results = True

# Main
input_type = 'text'  # draft, text  # TODO audio, video, fix_structures, none
output_types = {'draft', 'text', 'audio', 'subs', 'video'}  # draft, text, audio, video, subs  # TODO csv, csv_total
presaved = {}  # translation, mfa, audio  # reuse processed data from a previous run. Use only with the same input text
# …Reusing audio for video can be performed if the previous generation included audio and subs
# …in this case don’t set audio and subs in output_types for current generation and set manual_subs to False.

ignore_warnings = False  # don’t stop if config warnings are detected

translation_provider = 'Argos'  # GoogleCloud, Argos  # TODO DeepL, Yandex
align_with_translation_file = False
untranslatable_entities = {'PERSON', 'PER', 'DATE'}  # PRODUCT, TIME?

lemma_intervals = tuple(b - a for a, b in pairwise(int(1.05**i) for i in range(200, 2 << 10)))
entity_intervals = tuple(b - a for a, b in pairwise(int(1.1**i) for i in range(200, 2 << 10)))

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
break_between_vocabulary_ms = 400
break_in_vocabulary_ms = 100

# # for the "long" synthesis (>5000b for segment) with GoogleCloud
google_cloud_project_id = getenv('GCP_ID')
google_cloud_project_location = getenv('GCP_location')

# Audio alignment. To reuse words from a synthesized sentence in a vocabulary or visual alignment on the video
crossfade_ms = 50  # on the borders of each entity in vocabulary
final_silence_of_sentences_ms = 400  # crop final silence for the last word of the sentences
start_shift_ms = 5  # to capture a larger (or smaller) segments
end_shift_ms = 7
use_mfa = False  # the alternative is ssml=2. Don't use them together
mfa_dir = getenv('mfa_path')  # …/MFA/pretrained_models/
mfa_num_jobs = int((cpu_count() or 1) // 1.2)  # you can set your own value as an integer

use_ssml = 2  # currently only works with GC
# 0 – False; 1 – standard; 2 – with timestamps (to reuse pronunciation and more correct video captions)
reuse_synthesized = True  # works correctly only when use_mfa=True xor use_ssml=2

# ssml=1 with mfa=False and reuse=False – is the only case when synthesis is performed for the whole text at once

ssml_vocabulary_pitch = '+10%'  # '' | '{sign}{int}%' | '{int}st' semitones | 'x-low'/'low'/'medium'/'high'/'x-high'
ssml_vocabulary_volume = '-5dB'  # '' | '{sign}{int}dB' | 'silent'/'x-soft'/'soft'/'medium'/'loud'/'x-loud'/'default'
# works only with reuse_synthesized=False and use_ssml in {1, 2}
# if you set the vocabulary speed to 1 and both pitch and volume to '', the <prosody> tag in ssml for vocabulary
# …will be ommited; same for sentence speed only in ssml for sentences. Can be used to limit economy

# Video
manual_subs = False  # False = apply .ass file as hardsub on a clean background using ffmpeg.
# …Otherwise set the text manually using moviepy. It is much longer and a bit curvier, but potentially more manageable
sub_colors = ('FFFFFF', '00FFFF', 'AAFFAA', 'AAAAFF')  # (main, pronounce hl, known words hl, untranslatable words hl)
# …BGR format! For the last two you can set None to disable the highlighting of known/untranslatable words in general
highlight_after_main_sentence = False  # highlight known/untranslatable words only after the main sentence is spoken
video_width = 1920  # height is automatically adjusted
caption_font = 'C:/Windows/Fonts/arial.ttf'
font_size = 48
bottom_margin = 30
margin_between_original_and_translation = 50  # for manual subs only. with ffmpeg we use a line with half of font_size
line_height = 30  # for manual subs only


# _Inner
stop_words = set(stopwords.words(source_full_lang)) | set(stopwords.words(target_full_lang))
root_dir = Path(__file__).resolve().parent.parent
