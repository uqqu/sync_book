import logging

import config
import regex as re
import spacy
import torch
from transformers import AutoModel, AutoTokenizer


class TextProcessing:
    def __init__(self) -> None:
        self.source_lang = config.source_lang

        for model in (config.source_model, config.target_model):
            if not spacy.util.is_package(model):
                spacy.cli.download(model)

        self.nlp_src = spacy.load(config.source_model)
        self.nlp_trg = spacy.load(config.target_model)
        for nlp in (self.nlp_src, self.nlp_trg):
            nlp.add_pipe('sentencizer', before='parser')
            nlp.get_pipe('sentencizer').punct_chars.update((';', '\n', '\n\n', '\n\n\n'))

        embedding_model_name = 'sentence-transformers/LaBSE'
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)

        for param in self.embedding_model.parameters():
            param.requires_grad = False

    def get_sentences(self, text: str, is_source: bool = True) -> list[str]:
        model = self.nlp_src if is_source else self.nlp_trg
        return [sentence.text for sentence in model(text).sents]

    def get_merged_tokens(self, sentence: str, lang: str) -> list['Token']:
        return self._merge_tokens(self.nlp_src(sentence) if lang == self.source_lang else self.nlp_trg(sentence))

    @staticmethod
    def _merge_tokens(doc: 'Doc') -> list['Token']:
        '''Union problematic [eng] spacy tokens for simplier synchronization with Bert tokens.

        Our goal is to make the spacy tokens larger than the Bert tokens,
            so that we can then fit the latter to their lengths with combined embeddings.
        '''
        spans_to_merge = []
        i = 0
        n = len(doc)
        while i < n:
            token = doc[i]
            cur_text = token.text.lower()
            next_text = doc[i + 1].text.lower() if i < n - 1 else ''

            if i < n - 1 and (
                (token.is_punct and doc[i + 1].is_punct)
                or (re.match(r'^\p{Sc}$', cur_text) and re.match(r'^\d{1,3}([., ]\d{2,3})*$', next_text))
                or (re.match(r'^\p{L}+$', cur_text) and re.match(r'^n[\'’]?t$', next_text))  # n't/nt
                or (re.match(r'^\p{L}+in$', cur_text) and next_text in {'\'', '’'})  # -ing reduction
                or (re.match(r"^(go(nn|tt)a|cannot|c['’]mon|y['’]all)$", f'{cur_text}{next_text}'))
                or (re.match(r'^[\'’](?!(em|cause|bout|til|fore|tis|twas|gin)\b)\p{L}+$', next_text))
            ):
                spans_to_merge.append(doc[i : i + 2])

            elif 0 < i < n - 1 and re.match(r'^\p{L}-\p{L}$', f'{doc[i - 1].text[-1]}{cur_text}{next_text[0]}'):
                start = i - 1
                end = i + 2
                while end < n - 1 and re.match(r'^-\p{L}$', f'{doc[end].text}{doc[end + 1].text[0]}'):
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

    def add_tokens_embeddings(self, sentence: str, spacy_tokens: list['Token']) -> None:
        '''Add embedding attribute to spacy tokens, using Bert.

        Bert subword-tokens are merged into a single spacy token with a mean embedding value.
        '''
        spacy_idx = {j: i for i, t in enumerate(spacy_tokens) for j in range(t.idx, t.idx + len(t.text))}

        tokenized_input = self.embedding_tokenizer(
            sentence, return_tensors='pt', padding=True, truncation=True, return_offsets_mapping=True
        )
        offsets = tokenized_input.pop('offset_mapping')[0]
        with torch.no_grad():
            model_output = self.embedding_model(**tokenized_input, output_hidden_states=True)

        embeddings = torch.mean(torch.stack(model_output.hidden_states[-4:]), dim=0).squeeze(0)
        bert_tokens = self.embedding_tokenizer.convert_ids_to_tokens(tokenized_input.input_ids[0])

        if config.embedding_preprocessing_center:
            mean_embedding = torch.mean(embeddings, dim=0)
            embeddings = embeddings - mean_embedding

        aggregators = {
            'averaging': Averaging,
            'maxpooling': MaxPooling,
            'minpooling': MinPooling,
            'attention': Attention,
        }
        aggregator = aggregators[config.embedding_aggregator.lower()](embeddings, len(bert_tokens))

        subword_count = 0
        current_idx = 0

        for i, token in enumerate(bert_tokens):
            if offsets[i][0] == offsets[i][1]:
                continue
            start, end = spacy_idx[offsets[i][0].item()], spacy_idx[offsets[i][1].item() - 1]

            if start != end:
                raise RuntimeError(f'Intersect error. Bert "{token}", {offsets[i]} (spacy: {start}, {end})')

            if start == current_idx:
                aggregator.append(i)
                subword_count += 1
                continue

            if subword_count:
                spacy_tokens[current_idx]._.embedding = aggregator.get_final_embedding(subword_count)

            aggregator.start_new_word(i)
            subword_count = 1
            current_idx = start

        if subword_count:
            spacy_tokens[current_idx]._.embedding = aggregator.get_final_embedding(subword_count)


class BaseAggregator:
    def __init__(self, embeddings, n):
        self.embeddings = embeddings
        self.n = n
        self.current_embeddings = []

    def start_new_word(self, idx):
        '''Initialize a new word.'''
        self.current_embeddings = [self.embeddings[idx]]

    def append(self, idx):
        '''Add next subtoken.'''
        self.current_embeddings.append(self.embeddings[idx])

    def get_final_embedding(self, subword_count):
        '''Create final embedding by averaging all current embeddings.'''
        return torch.mean(torch.stack(self.current_embeddings), dim=0)


class Averaging(BaseAggregator):
    pass  # no overrides needed


class MaxPooling(BaseAggregator):
    def get_final_embedding(self, _):
        return torch.max(torch.stack(self.current_embeddings), dim=0).values


class MinPooling(BaseAggregator):
    def get_final_embedding(self, _):
        return torch.min(torch.stack(self.current_embeddings), dim=0).values


class Attention(BaseAggregator):
    def __init__(self, embeddings, n):
        super().__init__(embeddings, n)
        self.attention_scores = torch.zeros(self.n, dtype=torch.float32)

    def start_new_word(self, idx):
        super().start_new_word(idx)
        self.attention_scores = torch.zeros(self.n, dtype=torch.float32)
        self.attention_scores[0] = torch.norm(self.embeddings[idx], p=2)

    def append(self, idx):
        super().append(idx)
        attention_score = torch.matmul(self.embeddings[idx], torch.mean(torch.stack(self.current_embeddings), dim=0))
        l2_norm = torch.norm(self.embeddings[idx], p=2)
        self.attention_scores[len(self.current_embeddings) - 1] = attention_score * l2_norm

    def get_final_embedding(self, subword_count):
        '''Create final embedding weighted by attention_scores.'''
        if torch.sum(self.attention_scores) != 0:
            attention_weights = torch.nn.functional.softmax(self.attention_scores[:subword_count], dim=0)
            return torch.sum(attention_weights.unsqueeze(1) * torch.stack(self.current_embeddings), dim=0)
        return super().get_final_embedding(subword_count)
