import logging

import spacy
import torch
from spacy.tokens import Doc, Token
from transformers import AutoModel, AutoTokenizer

import config


class TextProcessing:
    def __init__(self) -> None:
        self.source_lang = config.source_lang

        self.nlp_src = spacy.load(config.source_model)
        self.nlp_trg = spacy.load(config.target_model)
        for nlp in (self.nlp_src, self.nlp_trg):
            nlp.add_pipe('sentencizer', before='parser')
            nlp.get_pipe('sentencizer').punct_chars.update((';', '\n', '\n\n', '\n\n\n'))

        embedding_model_name = 'bert-base-multilingual-cased'
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)

        for param in self.embedding_model.parameters():
            param.requires_grad = False

    def get_sentences(self, text: str, is_source: bool = True) -> list[str]:
        model = self.nlp_src if is_source else self.nlp_trg
        return [sentence.text for sentence in model(text).sents]

    def get_tokens(self, sentence: str, is_source: bool) -> Doc:
        return self.nlp_src(sentence) if is_source else self.nlp_trg(sentence)

    def get_merged_tokens(self, sentence: str, lang: str) -> list[Token]:
        return self._merge_tokens(self.get_tokens(sentence, lang == self.source_lang), sentence)

    @staticmethod
    def _merge_tokens(doc: Doc, original_text: str) -> list[Token]:
        '''Union problematic [eng] spacy tokens for simplier synchronization with Bert tokens.

        Our goal is to make the spacy tokens larger than the Bert tokens,
            so that we can then fit the latter to their lengths with combined embeddings.
        '''
        spans_to_merge = []
        i = 0
        n = len(doc)
        while i < n:
            token = doc[i]
            if ('\'' in token.text or '’' in token.text) and i > 0:
                spans_to_merge.append(doc[i - 1 : i + 1])
            elif i < n - 1 and (
                token.text == 'can' and doc[i + 1].text == 'not' or token.is_punct and doc[i + 1].is_punct
            ):
                spans_to_merge.append(doc[i : i + 2])
            elif token.text == '-' and 0 < i < n - 1:
                if original_text[token.idx - 1].isalpha() and original_text[token.idx + 1].isalpha():
                    start = i - 1
                    end = i + 2
                    while end < n and doc[end].text == '-' and original_text[doc[end].idx + 1].isalpha():
                        end += 2
                    spans_to_merge.append(doc[start:end])
                    logging.debug(f'Adding span to merge (complex hyphenated chain): {doc[start:end]}')
                    i = end - 1
            i += 1
        filtered_spans = spacy.util.filter_spans(spans_to_merge)
        with doc.retokenize() as retokenizer:
            for span in filtered_spans:
                retokenizer.merge(span)
        return [token for token in doc if '\n' not in token.text]

    def add_tokens_embeddings(self, sentence: str, spacy_tokens: list[Token]) -> None:
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

        # # averaging
        embeddings = torch.mean(torch.stack(model_output.hidden_states[-4:]), dim=0).squeeze(0)
        # # max-pool
        # hidden_states = torch.stack(model_output.hidden_states[-4:])
        # embeddings = torch.max(hidden_states, dim=0)[0].squeeze(0)
        # # attention
        # attentions = model_output.attentions[-1]
        # …
        # # l2 norm
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # # centering  # TODO?
        # mean_embedding = torch.mean(embeddings, dim=0)
        # embeddings = embeddings - mean_embedding

        bert_tokens = self.embedding_tokenizer.convert_ids_to_tokens(tokenized_input.input_ids[0])
        subword_count = 0
        current_embedding = torch.zeros_like(embeddings[0])
        current_idx = 0
        for i, token in enumerate(bert_tokens):
            if offsets[i][0] == offsets[i][1]:
                continue
            start, end = spacy_idx[offsets[i][0].item()], spacy_idx[offsets[i][1].item() - 1]
            if start != end:
                raise RuntimeError(f'Intersect error. Bert "{token}", {offsets[i]} (spacy: {start}, {end})')
            if start == current_idx:
                current_embedding += embeddings[i]
                subword_count += 1
                continue
            if subword_count:
                spacy_tokens[current_idx]._.embedding = current_embedding / subword_count
            current_embedding = embeddings[i]
            subword_count = 1
            current_idx = start
        if subword_count:
            spacy_tokens[current_idx]._.embedding = current_embedding / subword_count
