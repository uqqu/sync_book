import logging
import warnings
from collections import defaultdict

import numpy as np
from spacy.tokens import Token

from _structures import Entity, Lemma

warnings.filterwarnings('ignore', category=FutureWarning)


class TokenAligner:
    def __init__(self, container, sentence_obj):
        self.aligner = container.aligner
        self.config = container.config
        self.lemma_trie = container.lemma_trie
        self.lemmas = container.lemmas
        self.sentence_obj = sentence_obj
        self.tokens_src = sentence_obj.tokens_src
        self.tokens_trg = sentence_obj.tokens_trg
        self.src_to_trg, self.trg_to_src = self._align_tokens()

    def _align_tokens(self) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        '''Get third-party pairwise alignment and revert it to cross-reference.'''
        align = self.aligner.get_word_aligns(
            [t.text for t in self.tokens_src], [t.text for t in self.tokens_trg]
        )
        logging.info(f'Alignment: {align}')
        src_to_trg = defaultdict(list)
        trg_to_src = defaultdict(list)
        for a, b in align['inter']:
            src_to_trg[a].append(b)
            trg_to_src[b].append(a)
        return src_to_trg, trg_to_src

    def process_alignment(self, idx_src: int) -> None:
        '''Treat token with detected align-relation.'''
        if idx_src not in self.src_to_trg:
            self.add_not_aligned_token(idx_src)
            return
        to_trg_len = len(self.src_to_trg[idx_src])
        to_src_len = len(self.trg_to_src[self.src_to_trg[idx_src][0]])
        if to_trg_len == 1 and to_src_len == 1:
            self.one_to_one(idx_src)
        elif to_trg_len == 1 and to_src_len > 1:
            self.many_to_one(idx_src)
        elif to_trg_len > 1 and all(len(self.trg_to_src[x]) == 1 for x in self.src_to_trg[idx_src]):
            self.one_to_many(idx_src)
        else:
            self.many_to_many(idx_src)

    def add_not_aligned_token(self, idx_src: int) -> None:
        '''Try to manual add pair (one-to-one) for unaligned token.'''
        best_match, best_score = self._find_best_unaligned_match(self.tokens_src[idx_src])
        if best_match is not None:
            text_src = self.tokens_src[idx_src].text.lower()
            text_trg = self.tokens_trg[best_match].text.lower()
            if best_score < self.config.manual_min_align_weight:
                self.sentence_obj.possible_result.append((text_src, text_trg, best_score))
                logging.debug(
                    f'Not accepted not aligned. Found align {text_src} - {text_trg} with score {best_score}'
                )
            else:
                if ans := self._add_entity_by_lemma(idx_src, best_match):
                    self.sentence_obj.result.append(ans)
                    logging.debug(
                        f'Accepted not aligned. Found align {text_src} - {text_trg} with score {best_score}'
                    )
                self.sentence_obj.skip.add(self.tokens_src[idx_src])

    def _find_best_unaligned_match(self, token_src: Token) -> tuple[int | None, float]:
        best_match, best_score = None, 0
        for i, token_trg in enumerate(self.tokens_trg):
            if i in self.trg_to_src:
                continue
            score = self._cosine_similarity(token_src._.embedding, token_trg._.embedding)
            if score > best_score:
                best_score = score
                best_match = i
        return best_match, best_score

    @staticmethod
    def _cosine_similarity(x: 'torch.Tensor', y: 'torch.Tensor') -> 'np.float64':
        '''Calculates the cosine similarity between two embedding vectors.'''
        x = np.array(x)
        y = np.array(y)
        dot_product = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        return dot_product / (norm_x * norm_y + 1e-8)

    def _add_entity_by_lemma(self, idx_src, idx_trg, lemma_trg='', text_trg='') -> tuple[str, str] | bool:
        '''Check approved entity and lemma (add if not exist) by the distance rule and update it if possible.'''
        if not lemma_trg:
            lemma_trg = self.tokens_trg[idx_trg].lemma_
            text_trg = self.tokens_trg[idx_trg].text.lower()
        lemma_src = self.tokens_src[idx_src].lemma_
        text_src = self.tokens_src[idx_src].text.lower()

        key = f'{lemma_src}-{lemma_trg}'
        current_pos = self.sentence_obj.current_pos

        if key not in self.lemmas:
            self.lemmas[key] = Lemma(None, self.config.lemma_intervals)
        lemma_obj = self.lemmas[key]
        if text_src not in lemma_obj.children:
            ent_obj = Entity(None, self.config.entity_intervals, text_trg)
            lemma_obj.children[text_src] = ent_obj
        if lemma_obj.check_repeat(current_pos) and lemma_obj.children[text_src].check_repeat(current_pos):
            lemma_obj.children[text_src].update(current_pos)
            lemma_obj.update(current_pos)
            transl = lemma_obj.children[text_src].translation
            return text_src, (transl if transl else text_trg)
        return False

    def one_to_one(self, idx_src: int) -> None:
        '''Process a source token with a single reference to target token.'''
        idx_trg = self.src_to_trg[idx_src][0]
        score = self._cosine_similarity(
            self.tokens_src[idx_src]._.embedding, self.tokens_trg[idx_trg]._.embedding
        )
        if score < self.config.min_align_weight:
            logging.debug(f'O2O rejected with {score}')
            return
        logging.debug('O2O approved')
        if ans := self._add_entity_by_lemma(idx_src, idx_trg):
            self.sentence_obj.result.append(ans)

    def one_to_many(self, idx_src: int) -> None:
        '''Process a source token that references multiple target tokens.'''
        seq_tokens = self._filter_sequential(
            self.src_to_trg[idx_src], self.tokens_trg, self.tokens_src[idx_src]._.embedding
        )
        if not seq_tokens:
            logging.debug('Empty sequence of tokens for O2M')
            return
        text_trg = ' '.join(token.text.lower() for token in seq_tokens)
        lemma_trg = ' '.join(token.lemma_ for token in seq_tokens)
        if ans := self._add_entity_by_lemma(idx_src, 0, lemma_trg=lemma_trg, text_trg=text_trg):
            logging.debug('Approved O2M')
            self.sentence_obj.result.append(ans)

    def _add_entity_by_multilemma(self, seq_tokens_src: list[Token], seq_tokens_trg: list[Token]) -> None:
        '''Check approved entity (add it if not exist) by the distance rule and update it if possible.'''
        text_trg = ' '.join(token.text.lower() for token in seq_tokens_trg)
        ent_obj = Entity(None, self.config.entity_intervals, text_trg)
        entity = self.lemma_trie.add([token.lemma_ for token in seq_tokens_src], ent_obj)
        if entity.check_repeat(self.sentence_obj.current_pos):
            logging.debug(
                f'Accepted by repeat distance: {" ".join(x.text for x in seq_tokens_src)} {text_trg}'
            )
            entity.update(self.sentence_obj.current_pos)
            self.sentence_obj.result.append((' '.join(x.text for x in seq_tokens_src), text_trg))
        else:
            logging.debug(
                f'Rejected by repeat distance: {" ".join(x.text for x in seq_tokens_src)} {text_trg}'
            )
        self.sentence_obj.skip |= set(seq_tokens_src)

    def many_to_one(self, idx_src: int) -> None:
        '''Process a source token chain with a single reference to target token.'''
        trg_idx = self.src_to_trg[idx_src][0]
        seq_tokens_src = self._filter_sequential(
            self.trg_to_src[trg_idx], self.tokens_src, self.tokens_trg[trg_idx]._.embedding
        )
        if seq_tokens_src:
            self._add_entity_by_multilemma(seq_tokens_src, [self.tokens_trg[trg_idx]])

    def many_to_many(self, idx_src: int) -> None:
        '''Process a source token chain that references multiple target tokens.'''
        trg_idx = self.src_to_trg[idx_src][0]
        seq_tokens_src = self._filter_sequential(
            self.trg_to_src[trg_idx], self.tokens_src, self.tokens_trg[trg_idx]._.embedding
        )
        seq_tokens_trg = self._filter_sequential(
            self.src_to_trg[idx_src], self.tokens_trg, self.tokens_src[idx_src]._.embedding
        )
        if seq_tokens_src and seq_tokens_trg:
            self._add_entity_by_multilemma(seq_tokens_src, seq_tokens_trg)

    def _filter_sequential(self, aligned: list[int], tokens: list[Token], embedding) -> list[Token] | bool:
        '''Get support point from multiple alignment and select only sequential weight-approved tokens.'''
        s_aligned = set(aligned)
        best_match, best_score = None, 0
        for token_idx in aligned:
            score = self._cosine_similarity(tokens[token_idx]._.embedding, embedding)
            if score > best_score:
                best_score = score
                best_match = token_idx
        if best_match is None or best_score < self.config.min_align_weight:
            logging.debug(
                f'Best token {tokens[best_match] if best_match else "None"} have score {best_score}. Not accepted.'
            )
            return False
        seq_tokens = self._get_token_sequence(tokens, best_match, s_aligned)
        return seq_tokens

    def _get_token_sequence(self, tokens: list[Token], best_match: int, s_aligned: set[int]) -> list[Token]:
        '''Select only sequential weight-approved tokens.'''
        seq_tokens = [tokens[best_match]]
        for i, left_token in enumerate(tokens[:best_match][::-1], start=1):
            if left_token.is_punct:
                continue
            if best_match - i not in s_aligned:
                break
            seq_tokens.insert(0, left_token)
        for i, right_token in enumerate(tokens[best_match + 1 :], start=1):
            if right_token.is_punct:
                continue
            if best_match + i not in s_aligned:
                break
            seq_tokens.append(right_token)
        return seq_tokens
