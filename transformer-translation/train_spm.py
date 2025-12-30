import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="en.txt,fr.txt",
    model_prefix="spm",
    vocab_size=115,
    model_type="bpe",
    character_coverage=1.0
)

print("SentencePiece model trained successfully!")
