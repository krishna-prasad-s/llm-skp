import urllib.request
from tokeniser import get_tokens, generate_vocabulary, SimpleTokenizerV1, SimpleTokenizerV2
import tiktoken

url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open(file_path, "r") as f:
    text = f.read()
print("len:", len(text))


tokenised_text = get_tokens(text)
print("len:", len(tokenised_text))

vocab = generate_vocabulary(text)

tokeniser = SimpleTokenizerV1(vocab)
test = "Gisburn had given up his painting!"
ids = tokeniser.encode(test)
print(ids)
decoded = tokeniser.decode(ids)
print(decoded)
eot = "<|endoftext|>"
tknzr2 = SimpleTokenizerV2(vocab)
txt1 = "hello do you like to have chai,"
txt2 = "in the good supermorning"
jt = eot.join([txt1, txt2])
print(jt)
print(tknzr2.decode(tknzr2.encode(jt)))

ttk = tiktoken.get_encoding("gpt2")
ttk_encodings = ttk.encode(jt, allowed_special={eot})
print(ttk_encodings)
print(ttk.decode(ttk_encodings))
