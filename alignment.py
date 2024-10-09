import logging
import warnings
from collections import defaultdict

import numpy as np
from spacy.tokens import Token

warnings.filterwarnings('ignore', category=FutureWarning)


class TokenAligner:
    def __init__(self, container, tokens_src, tokens_trg):
        self.aligner = container.aligner
        self.tokens_src = tokens_src
        self.tokens_trg = tokens_trg
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

    def process_alignment(self, idx_src: int) -> tuple[float, list[Token], list[Token]]:
        '''Treat token with detected align-relation.'''
        if idx_src not in self.src_to_trg:
            return self.treat_not_aligned_token(idx_src)
        to_trg_len = len(self.src_to_trg[idx_src])
        to_src_len = len(self.trg_to_src[self.src_to_trg[idx_src][0]])
        if to_trg_len == 1 and to_src_len == 1:
            return self.one_to_one(idx_src)
        if to_trg_len == 1 and to_src_len > 1:
            return self.many_to_one(idx_src)
        if to_trg_len > 1 and all(len(self.trg_to_src[x]) == 1 for x in self.src_to_trg[idx_src]):
            return self.one_to_many(idx_src)
        return self.many_to_many(idx_src)

    def _find_best_match(self, checkable_tokens: dict[int, Token], control_token: Token) -> tuple[int, float]:
        best_match_idx, best_score = 0, 0.0
        for idx, token in checkable_tokens.items():
            if token.is_punct:
                continue
            score = self._cosine_similarity(token._.embedding, control_token._.embedding)
            if score > best_score:
                best_score = score
                best_match_idx = idx
        return best_match_idx, best_score

    @staticmethod
    def _cosine_similarity(x: 'torch.Tensor', y: 'torch.Tensor') -> float:
        '''Calculate the cosine similarity between two embedding vectors.'''
        x = np.array(x)
        y = np.array(y)
        dot_product = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        return float(dot_product / (norm_x * norm_y + 1e-8))

    @staticmethod
    def _get_token_sequence(tokens: dict[int, Token], idx: int) -> list[Token]:
        '''Select only sequential tokens.'''
        if not tokens:
            return []
        seq_tokens = [tokens[idx]]
        for i in range(idx - 1, -1, -1):
            if i not in tokens:
                break
            if tokens[i].is_punct:  # TODO TODO TODO
                continue
            seq_tokens.insert(0, tokens[i])
        for i in range(idx + 1, int(max(tokens.keys())) + 1):
            if i not in tokens:
                break
            if tokens[i].is_punct:
                continue
            seq_tokens.append(tokens[i])
        return seq_tokens

    def treat_not_aligned_token(self, idx_src: int) -> tuple[float, list[Token], list[Token]]:
        '''Try to manual add pair (one-to-one only) for unaligned token.'''
        unaligned_trg_tokens = {i: t for i, t in enumerate(self.tokens_trg) if i not in self.trg_to_src}
        best_match_idx, best_score = self._find_best_match(unaligned_trg_tokens, self.tokens_src[idx_src])
        logging.debug(f'Unaligned with {best_score}')
        return best_score, [self.tokens_src[idx_src]], [self.tokens_trg[best_match_idx]]

    def one_to_one(self, idx_src: int) -> tuple[float, list[Token], list[Token]]:
        '''Process a source token with a single reference to target token.'''
        idx_trg = self.src_to_trg[idx_src][0]
        score = self._cosine_similarity(
            self.tokens_src[idx_src]._.embedding, self.tokens_trg[idx_trg]._.embedding
        )
        logging.debug(f'O2O with {score}')
        return score, [self.tokens_src[idx_src]], [self.tokens_trg[idx_trg]]

    def one_to_many(self, idx_src: int) -> tuple[float, list[Token], list[Token]]:
        '''Process a source token that references multiple target tokens.'''
        checkable_tokens = {idx_trg: self.tokens_trg[idx_trg] for idx_trg in self.src_to_trg[idx_src]}
        best_match_idx, best_score = self._find_best_match(checkable_tokens, self.tokens_src[idx_src])
        seq_tokens_trg = self._get_token_sequence(checkable_tokens, best_match_idx)
        logging.debug(f'O2M ({len(seq_tokens_trg)}) sequence with {best_score}')
        return best_score, [self.tokens_src[idx_src]], seq_tokens_trg

    def many_to_one(self, idx_src: int) -> tuple[float, list[Token], list[Token]]:
        '''Process a source token chain with a single reference to target token.'''
        idx_trg = self.src_to_trg[idx_src][0]
        checkable_tokens = {idx_src: self.tokens_src[idx_src] for idx_src in self.trg_to_src[idx_trg]}
        best_match_idx, best_score = self._find_best_match(checkable_tokens, self.tokens_trg[idx_trg])
        seq_tokens_src = self._get_token_sequence(checkable_tokens, best_match_idx)
        logging.debug(f'M2O ({len(seq_tokens_src)}) sequence with {best_score}')
        return best_score, seq_tokens_src, [self.tokens_trg[idx_trg]]

    def many_to_many(self, idx_src: int) -> tuple[float, list[Token], list[Token]]:
        '''Process a source token chain that references multiple target tokens.'''
        best_src, best_trg, best_score = 0, 0, 0.0
        for idx_trg in self.src_to_trg[idx_src]:
            checkable_tokens = {idx_src: self.tokens_src[idx_src] for idx_src in self.trg_to_src[idx_trg]}
            curr_match, curr_score = self._find_best_match(checkable_tokens, self.tokens_trg[idx_trg])
            if curr_score > best_score:
                best_src = curr_match
                best_trg = idx_trg
                best_score = curr_score

        checkable_tokens_src = {idx_src: self.tokens_src[idx_src] for idx_src in self.trg_to_src[best_trg]}
        checkable_tokens_trg = {idx_trg: self.tokens_trg[idx_trg] for idx_trg in self.src_to_trg[best_src]}
        seq_tokens_src = self._get_token_sequence(checkable_tokens_src, best_src)
        seq_tokens_trg = self._get_token_sequence(checkable_tokens_trg, best_trg)
        logging.debug(f'M2M ({len(seq_tokens_src)}, {len(seq_tokens_trg)}) with {best_score}')
        return best_score, seq_tokens_src, seq_tokens_trg
