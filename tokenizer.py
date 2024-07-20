import torch
from transformers import BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

class AdvancedTokenizer:
    def __init__(self, vocab_size):
        self.base_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tfidf = TfidfVectorizer(max_features=vocab_size)
        self.vocab_size = vocab_size

    def fit(self, texts):
        self.tfidf.fit(texts)
        self.word_importance = {word: score for word, score in zip(self.tfidf.get_feature_names(), self.tfidf.idf_)}

    def encode(self, text):
        tokens = self.base_tokenizer.tokenize(text)
        importance_scores = [self.word_importance.get(token, 1.0) for token in tokens]
        token_ids = self.base_tokenizer.convert_tokens_to_ids(tokens)
        return torch.tensor(token_ids), torch.tensor(importance_scores)

    def decode(self, token_ids):
        tokens = self.base_tokenizer.convert_ids_to_tokens(token_ids)
        return self.base_tokenizer.convert_tokens_to_string(tokens)