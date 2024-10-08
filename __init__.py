import warnings

from config import Config
from main import Main
from text_processing import TextProcessing
from translation import Translator

warnings.filterwarnings('ignore', category=FutureWarning)

__all__ = ['TextProcessing', 'Translator', 'Main']

config = Config()
text_processor = TextProcessing()
