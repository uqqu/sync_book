import regex as re

import config


class BaseNode:
    '''Parent for entities and lemmas.'''

    def __init__(self, intervals: tuple[int]) -> None:
        self.intervals = intervals
        self.last_pos = 0
        self.level = 0

    def check_repeat(self, position: int) -> bool:
        '''Check the necessary distance to repeat (reinforce) both the lemma and the wordform.'''
        if not self.level:
            return True
        return self.intervals[self.level] < position - self.last_pos

    def update(self, new_pos: int) -> None:
        '''Update repeat distance.'''
        self.level += 1
        self.last_pos = new_pos


class Entity(BaseNode):
    '''Leaf for LemmaDict and LemmaTrie that stores translation and it repeat distance.'''

    def __init__(self, translation: str) -> None:
        super().__init__(config.entity_intervals)
        self.translation = translation


class LemmaDict(BaseNode):
    '''Dict structure to store single lemmas with entity chlidren.

    Dict takes into account texts along with lemmas.
    Root contain second level 'LemmaDict' objects by combination both source and target lemmas,
        second level in turn contain 'Entity' objects by raw text, to separate word forms.
    As BaseNode successor it can be checked by repeat distance.
    '''

    def __init__(self) -> None:
        super().__init__(config.lemma_intervals)
        self.children: dict[str, LemmaDict | Entity] = {}

    def add(self, lemma_src: str, text_src: str, lemma_trg: str, text_trg: str) -> tuple['LemmaDict', Entity]:
        '''Add combined lemma and specific word with leaf 'Entity' for translation (w/o overwrite for all).'''
        lemma_key = f'{lemma_src}-{lemma_trg}'

        if lemma_key not in self.children:
            self.children[lemma_key] = LemmaDict()
        lemma_obj = self.children[lemma_key]
        ent_key = f'{text_src}-{text_trg}'
        if ent_key not in lemma_obj.children:
            lemma_obj.children[ent_key] = Entity(text_trg)

        return lemma_obj, lemma_obj.children[ent_key]

    def length(self) -> tuple[int, int]:
        '''Get count of lemmas and their entities.'''
        return len(self.children), sum(len(x.children) for x in self.children.values())


class LemmaTrie:
    '''Trie structure to store chain of lemmas for idioms and expressions.

    Path from root to leaf in trie takes in account only source lemmas.
    Leaf 'Entity' used only to store translation (always) and repeat distance.
    '''

    def __init__(self) -> None:
        self.children: dict[str, LemmaTrie | Entity] = {}

    def search(self, tokens: list['Token'], depth: int = 0, punct_tail: int = 0) -> tuple[Entity | None, int]:
        '''Search the deepest leaf by the given list of tokens, if it possible.

        Return 'Entity' and it depth from original list of tokens.
        '''
        if not tokens:
            return None, 0
        token = tokens.pop(0)
        if not re.match(config.word_pattern, token.text):
            return self.search(tokens, depth + 1, punct_tail + 1)
        if child := self.children.get(token.lemma_):
            child = child.search(tokens, depth + 1)
        return child if child and child[0] else (self.children.get('#'), depth - punct_tail)

    def add(self, lemmas: list[str], entity: Entity) -> Entity:
        '''Add a chain of lemmas with a leaf entity node w/o overwrite, return leaf entity (new or old).'''
        if not lemmas:
            if '#' not in self.children:
                self.children['#'] = entity
            return self.children['#']
        if lemmas[0] not in self.children:
            self.children[lemmas[0]] = LemmaTrie()
        return self.children[lemmas[0]].add(lemmas[1:], entity)

    def length(self) -> int:
        '''Get count of all leafs.'''
        return sum(child.length() if isinstance(child, LemmaTrie) else 1 for child in self.children.values())
