import regex as re

from spacy.tokens import Token

from config import Config


class BaseNode:
    def __init__(self, pos: int | None, intervals: list[int]) -> None:
        self.last_pos = pos
        self.intervals = intervals
        self.level = 0

    def check_repeat(self, pos: int) -> bool:
        if self.last_pos is None:
            return True
        return self.intervals[self.level] < pos - self.last_pos

    def update(self, new_pos: int) -> None:
        self.level += 1
        self.last_pos = new_pos


class Lemma(BaseNode):
    def __init__(self, pos: int | None, intervals: list[int]) -> None:
        super().__init__(pos, intervals)
        self.children: dict[str, 'Entity'] = {}


class Entity(BaseNode):
    def __init__(self, pos: int | None, intervals: list[int], translation: str) -> None:
        super().__init__(pos, intervals)
        self.translation = translation


class LemmaTrie:
    '''Trie structure to store chain of lemmas for idioms and expressions.'''
    punct_pattern = Config().punct_pattern

    def __init__(self) -> None:
        self.children: dict[str, 'LemmaTrie' | 'Entity'] = {}

    def search(self, tokens: list[Token], i: int, punct_tail: int = 0) -> tuple[Entity | None, int, int]:
        '''Return the deepest leaf by the given list of tokens, if it possible.'''
        if i >= len(tokens):
            return (None, 0, 0)
        if not re.match(LemmaTrie.punct_pattern, tokens[i].text):
            return self.search(tokens, i + 1, punct_tail + 1)
        if child := self.children.get(tokens[i].lemma_):
            child = child.search(tokens, i + 1)
        return child if child and child[0] else (self.children.get('#'), i, punct_tail)

    def add(self, lemmas: list[str], entity: Entity) -> Entity:
        '''Idempotently adds a chain of lemmas with a leaf entity node, if not exist, and return leaf.'''
        if not lemmas:
            if '#' not in self.children:
                self.children['#'] = entity
            return self.children['#']
        if lemmas[0] not in self.children:
            self.children[lemmas[0]] = LemmaTrie()
        return self.children[lemmas[0]].add(lemmas[1:], entity)
