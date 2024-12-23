import re

def get_tokens(text):
    tokenised_text = re.split(r'([,.:;?_!"()\']|--|\s)', text) # split by punctuation and whitespace
    tokenised_text = [i.strip() for i in tokenised_text if i.strip()] # remove empty strings
    return tokenised_text

def generate_vocabulary(text):
    tokenised_text = get_tokens(text)
    vocabulary = sorted(set(tokenised_text))
    vocabulary.extend(["<|endoftext|>", "<|unk|>"])
    int_vocab = {token: integer for integer, token in enumerate(vocabulary)}
    return int_vocab
    

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        tokenised_text = get_tokens(text)
        ids = [self.str_to_int[s] for s in tokenised_text]
        return ids
        
    def decode(self, ids):
        decoded_text = " ".join([self.int_to_str[i] for i in ids])
        decoded_text = re.sub(r'\s([,.:;?_!"()\'])', r'\1', decoded_text)
        return decoded_text

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        tokenised_text = get_tokens(text)
        tokenised_text = [token if token in self.str_to_int else "<|unk|>" for token in tokenised_text]
        ids = [self.str_to_int[s] for s in tokenised_text]
        return ids
        
    def decode(self, ids):
        decoded_text = " ".join([self.int_to_str[i] for i in ids])
        decoded_text = re.sub(r'\s([,.:;?_!"()\'])', r'\1', decoded_text)
        return decoded_text