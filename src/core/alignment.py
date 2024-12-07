import warnings
from collections import defaultdict
from itertools import dropwhile

import config
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)


class TokenAligner:
    '''Align tokens for two languages by their embeddings.'''
    def __init__(self, tokens_src: list['Token'], tokens_trg: list['Token']) -> None:
        self.tokens_src = tokens_src
        self.tokens_trg = tokens_trg
        self.alignment = np.zeros((len(self.tokens_src), len(self.tokens_trg)))
        for i, src_token in enumerate(self.tokens_src):
            for j, trg_token in enumerate(self.tokens_trg):
                self.alignment[i, j] = self._cosine_similarity(src_token._.embedding, trg_token._.embedding)
        self.src_to_trg = {}
        self.trg_to_src = defaultdict(list)
        self.construct_pairs()
        for i, v in self.src_to_trg.items():
            for j in v:
                self.trg_to_src[j].append(i)
        self.src_to_trg = {i: sorted(v) for i, v in self.src_to_trg.items()}
        self.trg_to_src = {i: sorted(v) for i, v in self.trg_to_src.items()}

    @staticmethod
    def _cosine_similarity(x: 'torch.Tensor', y: 'torch.Tensor') -> float:
        '''Calculate the cosine similarity between two embedding vectors.'''
        if x is None or y is None:
            return 0.0
        x = x.detach().numpy()
        y = y.detach().numpy()
        dot_product = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        return float(dot_product / (norm_x * norm_y + 1e-8))

    def construct_pairs(self) -> None:
        '''Recursively find neighbouring pairs for tokens in the same dimension.'''
        start_i, start_j = np.unravel_index(np.argmax(self.alignment), self.alignment.shape)
        if self.alignment[start_i, start_j] < config.min_align_weight:
            return
        src_tokens = [start_i]
        trg_tokens = [start_j]
        i, j = start_i, start_j
        for si in (1, -1):
            while 0 < i < len(self.tokens_src) - 1:
                i += si
                if np.argmax(self.alignment[i, :]) != j or self.alignment[i, j] < config.min_align_weight:
                    break
                src_tokens.append(i)
            i = start_i
        self.alignment[start_i, :] = 0

        for sj in (1, -1):
            while 0 < j < len(self.tokens_trg) - 1:
                j += sj
                if np.argmax(self.alignment[:, j]) != i or self.alignment[i, j] < config.min_align_weight:
                    break
                trg_tokens.append(j)
            j = start_j
        self.alignment[:, start_j] = 0

        self._append_pairs(src_tokens, trg_tokens)
        self.construct_pairs()

    def _append_pairs(self, src_tokens: list[int], trg_tokens: list[int]) -> None:
        '''Convert pairs of tokens to groups and save it.'''
        l1 = lambda tokens: lambda t: 'Art' in tokens[t].morph.get('PronType')
        l2 = lambda tokens: lambda t: 'Art' in tokens[t].morph.get('PronType') or tokens[t].is_punct
        src_tokens = list(dropwhile(l1(self.tokens_src), list(dropwhile(l2(self.tokens_src), src_tokens[::-1]))[::-1]))
        trg_tokens = list(dropwhile(l1(self.tokens_trg), list(dropwhile(l2(self.tokens_trg), trg_tokens[::-1]))[::-1]))
        if not src_tokens or not trg_tokens:
            return

        cur_obj = None
        for i in src_tokens:
            if i in self.src_to_trg:
                cur_obj = self.src_to_trg[i]
                break

        if cur_obj is None:
            cur_obj = set(trg_tokens)
        else:
            cur_obj |= set(trg_tokens)

        for i in src_tokens:
            self.src_to_trg[i] = cur_obj

    def process_alignment(self, idx_src: int) -> tuple[list, list]:
        '''Return aligned tokens by source index.'''
        if idx_src not in self.src_to_trg:
            return [], []
        return (
            [self.tokens_src[i] for i in self.trg_to_src[self.src_to_trg[idx_src][0]]],
            [self.tokens_trg[i] for i in self.src_to_trg[idx_src]],
        )
