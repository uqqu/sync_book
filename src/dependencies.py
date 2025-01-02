from jinja2 import Environment, FileSystemLoader
from regex import sub

import config
from core._structures import StructureManager
from core.synthesis import SpeechSynthesizer
from core.text_preprocessing import TextPreprocessing
from core.translation import Translator


def __getattr__(name):
    if name in _cache:
        return _cache[name]
    _cache[name] = _factories[name]()
    return _cache[name]


def _create_templates():
    templates_dir = config.root_dir / 'src' / 'templates'
    names = set()
    for file_path in templates_dir.rglob('*.ssml'):
        if not file_path.name.endswith('.min.ssml'):
            minified_path = file_path.with_name(f'{file_path.stem}.min{file_path.suffix}')
            if not minified_path.exists():
                content = file_path.read_text(encoding='utf-8')
                minified_content = sub(r'\s*\n\s*', '', content)
                minified_path.write_text(minified_content, encoding='utf-8')
        names.add(file_path.stem.split('.')[0])

    templates_env = Environment(loader=FileSystemLoader(templates_dir))  # nosec
    return {name: templates_env.get_template(f'{name}.min.ssml') for name in names}


_cache: dict = {}

_factories = {
    'translator': Translator,
    'synthesizer': SpeechSynthesizer,
    'text_preprocessor': TextPreprocessing,
    'structures': StructureManager,
    'templates': _create_templates,
}
