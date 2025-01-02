import logging

import config
import regex as re
import spacy
import torch
from transformers import AutoModel, AutoTokenizer


class TextPreprocessing:
    def __init__(self) -> None:
        for model in (config.source_model, config.target_model):
            if not spacy.util.is_package(model):
                spacy.cli.download(model)

        self.src_nlp = spacy.load(config.source_model)
        self.trg_nlp = spacy.load(config.target_model)
        for nlp in (self.src_nlp, self.trg_nlp):
            nlp.add_pipe('sentencizer', before='parser')
            nlp.get_pipe('sentencizer').punct_chars.update((';', '\n', '\n\n', '\n\n\n'))

        embedding_model_name = 'sentence-transformers/LaBSE'
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)

        for param in self.embedding_model.parameters():
            param.requires_grad = False

    def get_sentences(self, text: str, is_source: bool = True) -> list[str]:
        '''Return list of sentences from the entire text with corrected endings.'''
        model = self.src_nlp if is_source else self.trg_nlp
        sentences = []
        tail = ''
        for sentence in model(text).sents:
            sentence = sentence.text
            match = re.search(r'\n\P{L}*$', sentence)
            if match:
                sentence = sentence[: match.start()]
                new_tail = match.group(0)
            else:
                new_tail = ''
            sentences.append(f'{tail}{sentence}')
            tail = new_tail

        return sentences

    def get_tokens_with_embeddings(self, src_text: str, trg_text: str) -> tuple[list['Token'], list['Token']]:
        '''Return normalized spacy tokens with additional contextual embeddings.'''
        src_tokens = self._get_merged_tokens(self.src_nlp(src_text))
        trg_tokens = self._get_merged_tokens(self.trg_nlp(trg_text))
        logging.info(f'{src_tokens=}, {trg_tokens=}')
        self._add_token_embeddings(src_text, src_tokens)
        self._add_token_embeddings(trg_text, trg_tokens)
        return src_tokens, trg_tokens

    def _get_merged_tokens(self, doc: 'Doc') -> list['Token']:
        '''Merge tokens by spaces in the original text for syncronization with Bert tokens.

        Universal way with simpliest logic, but typos in the original text may cause tokenization errors.
        '''
        spans_to_merge = []
        current_span = []

        for token in doc:
            if not token.is_punct or token.text in {'\'', '’', '-', '–', '—'}:
                current_span.append(token)
                if not token.whitespace_:
                    continue
            if len(current_span) > 1:
                spans_to_merge.append(doc[current_span[0].i : current_span[-1].i + 1])
            current_span = []

        if len(current_span) > 1:
            spans_to_merge.append(doc[current_span[0].i : current_span[-1].i + 1])

        filtered_spans = spacy.util.filter_spans(spans_to_merge)
        with doc.retokenize() as retokenizer:
            for span in filtered_spans:
                retokenizer.merge(span)

        return [token for token in doc if not token.is_space]

    def __get_merged_tokens(self, doc: 'Doc') -> list['Token']:
        '''Merge problematic [eng] spacy tokens for simpler synchronization with Bert tokens.'''
        spans_to_merge = []
        i = 0
        n = len(doc)
        while i < n:
            cur_text = doc[i].text.lower()
            next_text = doc[i + 1].text.lower() if i < n - 1 else ''
            pattern = re.compile(
                r'''
                (\p{L}+(?<!do))in['’]  # -ing reduction (only for unrecognized by spacy)
                | \p{L}+(?<!wa)n['’]?t  # n't/nt reduction
                | ['’](tain['’]t|n|til|fore|tis|twas|gin)  # leading apostrophe
                | (go(nn|tt)a|cannot|an['’]|\p{L}+(?<!wa)nt)  # exclusions
                | \p{Sc}\d{1,3}([., ]\d{2,3})*  # currency check (currency symbol + digits, possibly with separators)
                | \p{L}+(?<!in)['’](?!(em|cause|bout)\b)\p{L}+  # dialect reductions
            ''',
                re.X,
            )

            if i < n - 1 and (
                (doc[i].is_punct and doc[i + 1].is_punct) or (re.fullmatch(pattern, f'{cur_text}{next_text}'))
            ):
                spans_to_merge.append(doc[i : i + 2])

            elif 0 < i < n - 1 and re.fullmatch(r'\p{L}-\p{L}', f'{doc[i - 1].text[-1]}{cur_text}{next_text[0]}'):
                start = i - 1
                end = i + 2
                while end < n - 1 and re.fullmatch(r'-\p{L}', f'{doc[end].text}{doc[end + 1].text[0]}'):
                    end += 2
                spans_to_merge.append(doc[start:end])
                logging.debug(f'Adding span to merge (complex hyphenated chain): {doc[start:end]}')
                i = end - 1
            i += 1
        filtered_spans = spacy.util.filter_spans(spans_to_merge)
        with doc.retokenize() as retokenizer:
            for span in filtered_spans:
                retokenizer.merge(span)
        return [token for token in doc if not token.is_space]

    def _add_token_embeddings(self, sentence: str, spacy_tokens: list['Token']) -> None:
        '''Add embedding custom attribute to spacy tokens.

        Match the subword tokens from Bert with the merged spaCy tokens and aggregate these embeddings.
        '''
        spacy_idx = {j: i for i, t in enumerate(spacy_tokens) for j in range(t.idx, t.idx + len(t.text))}
        bert_tokens, offsets, embeddings = self._get_token_embeddings(sentence)

        if config.embedding_preprocessing_centering:
            mean_embedding = torch.mean(embeddings, dim=0)
            embeddings = embeddings - mean_embedding

        aggr = {'averaging': Averaging, 'maxpooling': MaxPooling, 'minpooling': MinPooling, 'attention': Attention}
        aggregator = aggr[config.embedding_aggregator.lower()](len(bert_tokens))

        subword_count = 0
        current_idx = 0

        for token, offset, embedding in zip(bert_tokens, offsets, embeddings):
            if offset[0] == offset[1]:
                continue
            start, end = spacy_idx[offset[0]], spacy_idx[offset[1] - 1]

            if start != end:
                raise RuntimeError(f'Intersect error. Bert "{token}", {offset} (spacy: {start}, {end})')

            if start == current_idx:
                aggregator.append(embedding)
                subword_count += 1
                continue

            if subword_count:
                spacy_tokens[current_idx]._.embedding = aggregator.get_final_embedding(subword_count)

            aggregator.start_new_word(embedding)
            subword_count = 1
            current_idx = start

        if subword_count:
            spacy_tokens[current_idx]._.embedding = aggregator.get_final_embedding(subword_count)

    def _get_token_embeddings(self, sentence: str) -> tuple[list[str], list[tuple[int, int]], list[torch.Tensor]]:
        '''Get the embeddings block by block (max 512 tokens in one block) and bring them back united.'''
        stride = 256
        max_length = 510
        all_tokens: list[str] = []
        all_offsets: list[tuple[int, int]] = []
        all_embeddings: list[torch.Tensor] = []

        tokens = self.embedding_tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = tokens['input_ids']

        for start in range(0, len(input_ids), stride):
            end = min(start + max_length, len(input_ids))
            block_input_ids = (
                [self.embedding_tokenizer.cls_token_id] + input_ids[start:end] + [self.embedding_tokenizer.sep_token_id]
            )
            with torch.no_grad():
                outputs = self.embedding_model(torch.tensor([block_input_ids]), output_hidden_states=True)
            hidden_states = torch.stack(outputs.hidden_states[-4:]).mean(dim=0).squeeze(0)

            all_tokens.extend(self.embedding_tokenizer.convert_ids_to_tokens(block_input_ids)[1:-1])
            all_offsets.extend(tokens['offset_mapping'][start:end])
            all_embeddings.extend(hidden_states[1:-1])

        return all_tokens, all_offsets, all_embeddings


class BaseAggregator:
    '''A basis for the aggregation of subwords embeddings when combining them into entire ones.'''

    def __init__(self, n: int) -> None:
        self.n: int = n
        self.current_embeddings: list[torch.Tensor] = []

    def start_new_word(self, embedding: torch.Tensor) -> None:
        '''Initialize a new word.'''
        self.current_embeddings = [embedding]

    def append(self, embedding: torch.Tensor) -> None:
        '''Add next subtoken.'''
        self.current_embeddings.append(embedding)

    def get_final_embedding(self, subword_count: int) -> torch.Tensor:
        '''Create final embedding by averaging all current embeddings.'''
        return torch.mean(torch.stack(self.current_embeddings), dim=0)


class Averaging(BaseAggregator):
    pass  # no overrides needed


class MaxPooling(BaseAggregator):
    def get_final_embedding(self, _) -> torch.Tensor:
        return torch.max(torch.stack(self.current_embeddings), dim=0).values


class MinPooling(BaseAggregator):
    def get_final_embedding(self, _) -> torch.Tensor:
        return torch.min(torch.stack(self.current_embeddings), dim=0).values


class Attention(BaseAggregator):
    def __init__(self, n: int) -> None:
        super().__init__(n)
        self.attention_scores = torch.zeros(self.n, dtype=torch.float32)

    def start_new_word(self, embedding: torch.Tensor) -> None:
        super().start_new_word(embedding)
        self.attention_scores = torch.zeros(self.n, dtype=torch.float32)
        self.attention_scores[0] = torch.norm(embedding, p=2)

    def append(self, embedding: torch.Tensor) -> None:
        super().append(embedding)
        attention_score = torch.matmul(embedding, torch.mean(torch.stack(self.current_embeddings), dim=0))
        l2_norm = torch.norm(embedding, p=2)
        self.attention_scores[len(self.current_embeddings) - 1] = attention_score * l2_norm

    def get_final_embedding(self, subword_count: int) -> torch.Tensor:
        '''Create final embedding weighted by attention_scores.'''
        if torch.sum(self.attention_scores) != 0:
            attention_weights = torch.nn.functional.softmax(self.attention_scores[:subword_count], dim=0)
            return torch.sum(attention_weights.unsqueeze(1) * torch.stack(self.current_embeddings), dim=0)
        return super().get_final_embedding(subword_count)
