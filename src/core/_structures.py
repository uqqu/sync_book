import logging
import pickle
from dataclasses import dataclass, field

import config
from regex import fullmatch
from spacy.tokens import Token


class StructureManager:
    def __init__(self):
        self.entity_counter = 0
        self.sentence_counter = 0
        self.lemma_dict = LemmaDict()
        self.lemma_trie = LemmaTrie()
        self.load_structures()

    def load_structures(self) -> None:
        '''Load structures and progress from a previously saved file.'''
        try:
            with open(config.root_dir / f'{config.storage_filename}.pkl', 'rb') as file:
                self.lemma_dict, self.lemma_trie, self.entity_counter, self.sentence_counter = pickle.load(file)
            len_dict = self.lemma_dict.length()
            logging.info(
                'Succesfully loaded previous user structures. '
                f'Dict {len_dict[0]} ({len_dict[1]}), Trie {self.lemma_trie.length()}.'
            )
            self.entity_counter = 0
        except FileNotFoundError:
            logging.info('Failed when attempting to load user structures.')

    def save_structures(self) -> None:
        '''Save structures and progress to a file for future reuse. Keep the previous version, if it exists.'''
        filepath = config.root_dir / f'{config.storage_filename}.pkl'
        if filepath.exists():
            old_filepath = filepath.with_suffix('.pkl.old')
            if old_filepath.exists():
                old_filepath.unlink()
            filepath.rename(old_filepath)

        with open(filepath, 'wb') as file:
            pickle.dump((self.lemma_dict, self.lemma_trie, self.entity_counter, self.sentence_counter), file)

    def get_pairs(self) -> list[tuple[str, str]]:
        '''Get text from leafs with their paths from trie and lemma level from dict structure.'''

        def get_trie_elems(elem: dict[str, LemmaTrie | Entity], path: list[str]) -> list[tuple[str, str]]:
            results = []
            for word, child in elem.items():
                if isinstance(child, LemmaTrie):
                    results.extend(get_trie_elems(child.children, path + [word]))
                else:
                    results.append((' '.join(path), f'{child.translation} ({child.level})'))
            return results

        trie_results = get_trie_elems(self.lemma_trie.children, [])

        dict_results = []
        for lemma, child in self.lemma_dict.children.items():
            src, trg = lemma.split('–—-')
            dict_results.append((src, f'{trg} ({child.level})'))
        return trie_results + dict_results


@dataclass
class UserToken:
    '''Custom token class for exporting from draft or transforming from spacy tokens.'''

    text: str
    lemma_: str
    index: int
    position: int
    is_punct: bool
    whitespace: bool
    audio: slice | int | None = None
    segments: list = field(default_factory=list)


class BaseNode:
    '''Parent for entities and lemmas.'''

    def __init__(self, intervals: tuple[int]) -> None:
        self.intervals = intervals
        self.last_pos = 0
        self.level = 0

    def check_repeat(self, position: int) -> bool:
        '''Check the necessary distance to repeat (reinforce) both the lemma and the word form.'''
        if not self.level:
            return True
        return self.intervals[self.level] < position - self.last_pos

    def update(self, new_pos: int) -> None:
        '''Update the repetition distance.'''
        if new_pos > self.last_pos:
            self.level += 1
            self.last_pos = new_pos


class Entity(BaseNode):
    '''A leaf for LemmaDict and LemmaTrie that stores translation and it repetition distance.'''

    def __init__(self, translation: str) -> None:
        super().__init__(config.entity_intervals)
        self.translation = translation.lower()


class LemmaDict(BaseNode):
    '''Dict structure for storing single lemmas with the entity chlidren.

    Dict consider texts along with lemmas.
    The root contains second level 'LemmaDict' objects by combination both source and target lemmas,
        the second level, in turn, contains 'Entity' objects by raw text, to separate word forms.
    As BaseNode successor it can be checked by repetition distance.
    '''

    def __init__(self) -> None:
        super().__init__(config.lemma_intervals)
        self.children: dict[str, LemmaDict | Entity] = {}

    def add(self, src: list[Token | UserToken], trg: list[Token | UserToken]) -> tuple['LemmaDict', Entity]:
        '''Add a combined lemma and a specific word with an 'Entity' leaf for translation (w/o overwrite for all).'''
        src_lemma = ' '.join(s.lemma_.lower() for s in src)
        src_text = ' '.join(s.text.lower() for s in src)
        trg_lemma = ' '.join(t.lemma_.lower() if t.lemma_ else '' for t in trg)
        trg_text = ' '.join(t.text.lower() for t in trg)

        lemma_key = f'{src_lemma}–—-{trg_lemma}'.lower()
        if lemma_key not in self.children:
            self.children[lemma_key] = LemmaDict()
        lemma_obj = self.children[lemma_key]

        ent_key = f'{src_text}–—-{trg_text}'.lower()
        if ent_key not in lemma_obj.children:  # type: ignore
            lemma_obj.children[ent_key] = Entity(trg_text)  # type: ignore

        return lemma_obj, lemma_obj.children[ent_key]  # type: ignore

    def length(self) -> tuple[int, int]:
        '''Get the count of lemmas and their entities.'''
        return len(self.children), sum(len(x.children) for x in self.children.values())  # type: ignore


class LemmaTrie:
    '''A trie structure for storing a chain of lemmas for idioms and expressions.

    The path from root to leaf in trie only considers the source lemmas.
    The 'Entity' leaf is only used to store the translation (always) and the repetition distance.
    '''

    def __init__(self) -> None:
        self.children: dict[str, LemmaTrie | Entity] = {}

    def search(self, tokens: list[Token | UserToken], depth=0, punct_tail=0) -> tuple[Entity | None, int]:
        '''Search for the deepest leaf on a given list of tokens, if possible.

        Return 'Entity' and its depth from the source token list.
        '''
        if not tokens:
            return self.children.get('#'), depth - punct_tail - 1  # type: ignore
        if tokens[0].is_punct:
            return self.search(tokens[1:], depth + 1, punct_tail + 1)
        if child := self.children.get(tokens[0].lemma_.lower()):
            child = child.search(tokens[1:], depth + 1)  # type: ignore
        return child if child and child[0] else (self.children.get('#'), depth - punct_tail)  # type: ignore

    def add(self, tokens: list[Token | UserToken], entity: Entity) -> Entity:
        '''Add a chain of lemmas with a leaf entity node w/o overwriting, return the leaf entity (new or old).'''
        if not tokens:
            if '#' not in self.children:
                self.children['#'] = entity
            return self.children['#']  # type: ignore
        if tokens[0].is_punct:
            return self.add(tokens[1:], entity)
        if (lemma := tokens[0].lemma_.lower()) not in self.children:
            self.children[lemma] = LemmaTrie()
        return self.children[lemma].add(tokens[1:], entity)  # type: ignore

    def length(self) -> int:
        '''Get the count of all the leaves.'''
        return sum(child.length() if isinstance(child, LemmaTrie) else 1 for child in self.children.values())
