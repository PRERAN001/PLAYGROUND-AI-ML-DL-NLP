import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("spm.model")

text = "hello world"
ids = sp.encode(text, out_type=int)

print("Token IDs:", ids)
print("Decoded:", sp.decode(ids))
