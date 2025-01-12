import logging
from collections import defaultdict
from itertools import dropwhile

import config
import dependencies as container
import numpy as np
from regex import fullmatch
from spacy.tokens import Token

from core._structures import Entity, UserToken


class TokenProcessing:
    def __init__(self, src_tokens: list[Token], trg_tokens: list[Token]) -> None:
        self.src_tokens = src_tokens
        self.trg_tokens = trg_tokens
        self.skip: set[int] = set()
        self.result: list[tuple[list[int], list[int], bool]] = []

    def process(self) -> tuple[list[UserToken], list[UserToken], list[tuple[list[int], list[int], bool]]]:
        '''Process all tokens by their type and return potential group pairs to add to the translation.'''
        for src_idx, src_token in enumerate(self.src_tokens):
            src_token._.position = container.structures.entity_counter
            container.structures.entity_counter += 1
            logging.debug(f'Processing token: {src_token}')
            if src_idx in self.skip:  # skip token if it has already been processed
                logging.debug('Skipping previously processed token')
            elif src_token.is_punct:  # skip punct w/o counting
                container.structures.entity_counter -= 1
                logging.debug('Skipping punctuation')
            elif src_token.ent_type_ in config.untranslatable_entities or 'Art' in src_token.morph.get('PronType'):
                logging.debug(
                    f'Skipping untranslatable entity or article: {src_token.ent_type_}'
                    f'– {src_token.morph.get("PronType")}'
                )
            elif self._trie_search_and_process(src_idx):  # check multiword origin chain
                logging.debug('Found multiword chain')
            elif self._add_named_entity(src_idx):  # treat named entity chain
                logging.debug('Named entity found and processed')
            elif src_token.text.lower() in config.stop_words:  # only after multiword check
                logging.debug('Skipping stopword')
            else:
                src_seq_idx, trg_seq_idx = TokenAligner(self.src_tokens, self.trg_tokens).process_alignment(src_idx)
                if src_seq_idx and trg_seq_idx:
                    self._append_aligned_tokens(src_seq_idx, trg_seq_idx)

        src_tokens = self._transform_tokens(self.src_tokens)
        trg_tokens = self._transform_tokens(self.trg_tokens)
        self.result = [
            ([src_tokens[i] for i in src], [trg_tokens[j] if isinstance(j, np.int64) else j for j in trg], status)
            for src, trg, status in self.result
        ]

        logging.debug(f'Result: {self.result}')
        return (src_tokens, trg_tokens, self.result)

    def _trie_search_and_process(self, idx: int) -> bool:
        '''Search for an existing source multiword sequence in LemmaTrie.'''
        entity, depth = container.structures.lemma_trie.search(self.src_tokens[idx:])
        if depth < 2:
            return False

        status = False
        if entity.check_repeat(self.src_tokens[idx]._.position):
            status = True
        trg = [
            UserToken(text=entity.translation, lemma_=None, index=None, position=None, is_punct=False, whitespace=True)
        ]
        self.result.append((list(range(idx, idx + depth)), trg, status))
        self.skip |= set(range(idx, idx + depth))
        logging.debug(f'Known multiword chain found: {" ".join(t.text for t in self.src_tokens[idx:idx+depth])}')
        return True

    def _add_named_entity(self, idx: int) -> bool:
        '''Add named entity without token alignment.'''
        seq_tokens = [self.src_tokens[idx]]
        for forw_token in self.src_tokens[idx + 1 :]:
            if forw_token.ent_iob_ != 'I':
                break
            seq_tokens.append(forw_token)
        if len(seq_tokens) < 2:
            return False

        translation = container.translator.translate(' '.join(s.text for s in seq_tokens))
        entity = container.structures.lemma_trie.add(seq_tokens, Entity(translation))
        trg = [
            UserToken(text=entity.translation, lemma_=None, index=None, position=None, is_punct=False, whitespace=True)
        ]
        self.result.append((list(range(idx, idx + len(seq_tokens))), trg, True))
        self.skip |= set(range(idx, idx + len(seq_tokens)))
        logging.debug(f'Named entity found and processed: {seq_tokens}')
        return True

    def _append_aligned_tokens(self, src_idxs: list[int], trg_idxs: list[int]) -> None:
        '''Add result token pairs with it output status (don’t update entity distance, it can be called for draft).'''
        status = True
        pos = self.src_tokens[src_idxs[0]]._.position
        src_group = [self.src_tokens[i] for i in src_idxs]
        if len(src_idxs) == 1:  # o2o/o2m  # new or existing dict entity
            lemma, entity = container.structures.lemma_dict.add(src_group, [self.trg_tokens[i] for i in trg_idxs])
            if not (lemma.check_repeat(pos) and entity.check_repeat(pos)):
                status = False
        else:  # m2m/m2o  # new trie entity
            container.structures.lemma_trie.add(src_group, Entity(' '.join(self.trg_tokens[i].text for i in trg_idxs)))
        self.result.append((src_idxs, trg_idxs, status))
        self.skip |= set(src_idxs)

    def _transform_tokens(self, tokens: list[Token]) -> list[UserToken]:
        '''Take necessary attributes from spacy tokens to highly controlled custom token structure.'''
        return [
            UserToken(
                text=t.text,
                lemma_=t.lemma_,
                index=t.idx,
                position=t._.position,
                is_punct=t.is_punct,
                whitespace=t.whitespace_,
            )
            for t in tokens
        ]


class TokenAligner:
    '''Align tokens for two languages by their embeddings.'''

    def __init__(self, src_tokens: list[Token], trg_tokens: list[Token]) -> None:
        self.src_tokens = src_tokens
        self.trg_tokens = trg_tokens
        src_embeddings = np.array([token._.embedding.detach().numpy() for token in self.src_tokens])
        trg_embeddings = np.array([token._.embedding.detach().numpy() for token in self.trg_tokens])
        self.alignment = self.cosine_similarity_matrix(src_embeddings, trg_embeddings)
        self.src_to_trg: dict[int, list | set] = {}
        self.trg_to_src: dict[int, list] = defaultdict(list)
        self.construct_pairs()
        self._normalize_alignment_mapping()

    @staticmethod
    def cosine_similarity_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''Calculate the cosine similarity matrix between two sets of embedding vectors.'''
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        y_norm = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)
        return np.dot(x_norm, y_norm.T)

    def construct_pairs(self) -> None:
        '''Recursively find neighbouring pairs for tokens in the same dimension.'''
        start_i, start_j = np.unravel_index(np.argmax(self.alignment), self.alignment.shape)
        if self.alignment[start_i, start_j] < config.min_align_weight:
            return
        src_group = [start_i]
        trg_group = [start_j]
        i, j = start_i, start_j
        for si in (1, -1):
            while 0 < i < len(self.src_tokens) - 1:
                i += si
                if np.argmax(self.alignment[i, :]) != j or self.alignment[i, j] < config.min_align_weight:
                    break
                src_group.append(i)
            i = start_i
        self.alignment[start_i, :] = 0

        for sj in (1, -1):
            while 0 < j < len(self.trg_tokens) - 1:
                j += sj
                if np.argmax(self.alignment[:, j]) != i or self.alignment[i, j] < config.min_align_weight:
                    break
                trg_group.append(j)
            j = start_j
        self.alignment[:, start_j] = 0

        self._append_pairs(sorted(src_group), sorted(trg_group))
        self.construct_pairs()

    def _append_pairs(self, src_group: list[int], trg_group: list[int]) -> None:
        '''Convert pairs of tokens to groups and save it.'''
        con = lambda tokens: lambda t: 'Art' in tokens[t].morph.get('PronType') or tokens[t].is_punct
        src_group = list(dropwhile(con(self.src_tokens), list(dropwhile(con(self.src_tokens), src_group))[::-1]))[::-1]
        trg_group = list(dropwhile(con(self.trg_tokens), list(dropwhile(con(self.trg_tokens), trg_group))[::-1]))[::-1]
        if not src_group or not trg_group:
            return

        cur_obj = None
        for i in src_group:
            if i in self.src_to_trg:
                cur_obj = self.src_to_trg[i]
                break

        if cur_obj is None:
            cur_obj = set(trg_group)
        else:
            cur_obj |= set(trg_group)

        for i in src_group:
            self.src_to_trg[i] = cur_obj

    def _normalize_alignment_mapping(self) -> None:
        '''Fill trg_to_src mapping and return temporary set values to lists.'''
        for i, v in self.src_to_trg.items():
            for j in v:
                self.trg_to_src[j].append(i)
        self.src_to_trg = {i: sorted(v) for i, v in self.src_to_trg.items()}
        self.trg_to_src = {i: sorted(v) for i, v in self.trg_to_src.items()}

    def process_alignment(self, src_idx: int) -> tuple[list, list]:
        '''Return aligned tokens by source index.'''
        if src_idx not in self.src_to_trg:
            return [], []
        return (self.trg_to_src[self.src_to_trg[src_idx][0]], self.src_to_trg[src_idx])
