import math
import torch
import torch.nn as nn
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
FFN_DIM = 256
BATCH_SIZE = 8
EPOCHS = 200
MAX_LEN = 50

sp = spm.SentencePieceProcessor()
sp.load("spm.model")

VOCAB_SIZE = sp.get_piece_size()
PAD_ID = sp.pad_id() if sp.pad_id() >= 0 else 0
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()


class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, sp):
        self.sp = sp
        self.src = open(src_file, encoding="utf-8").read().splitlines()
        self.tgt = open(tgt_file, encoding="utf-8").read().splitlines()

        min_len = min(len(self.src), len(self.tgt))
        self.src = self.src[:min_len]
        self.tgt = self.tgt[:min_len]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_ids = self.sp.encode(self.src[idx], out_type=int)
        tgt_ids = self.sp.encode(self.tgt[idx], out_type=int)

        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_batch = nn.utils.rnn.pad_sequence(
        src_batch, padding_value=PAD_ID
    )
    tgt_batch = nn.utils.rnn.pad_sequence(
        tgt_batch, padding_value=PAD_ID
    )

    return src_batch.to(DEVICE), tgt_batch.to(DEVICE)

dataset = TranslationDataset("en.txt", "fr.txt", sp)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[: x.size(0)]


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE, D_MODEL, padding_idx=PAD_ID
        )
        self.positional = PositionalEncoding(D_MODEL)

        self.transformer = nn.Transformer(
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_LAYERS,
            num_decoder_layers=NUM_LAYERS,
            dim_feedforward=FFN_DIM,
        )

        self.fc_out = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, src, tgt):
        src_emb = self.positional(self.embedding(src))
        tgt_emb = self.positional(self.embedding(tgt))

        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt.size(0)
        ).to(DEVICE)

        src_key_padding_mask = (src == PAD_ID).transpose(0, 1)
        tgt_key_padding_mask = (tgt == PAD_ID).transpose(0, 1)

        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        return self.fc_out(output)

model = TransformerModel().to(DEVICE)


criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for src, tgt in loader:
        optimizer.zero_grad()

        tgt_input = tgt[:-1]
        tgt_output = tgt[1:]

        logits = model(src, tgt_input)
        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            tgt_output.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")


def translate(sentence):
    model.eval()

    src_ids = [BOS_ID] + sp.encode(sentence, out_type=int) + [EOS_ID]
    src = torch.tensor(src_ids).unsqueeze(1).to(DEVICE)

    tgt = torch.tensor([[BOS_ID]]).to(DEVICE)

    for _ in range(MAX_LEN):
        logits = model(src, tgt)
        next_token = logits[-1].argmax(-1).item()
        tgt = torch.cat(
            [tgt, torch.tensor([[next_token]]).to(DEVICE)],
            dim=0,
        )
        if next_token == EOS_ID:
            break

    return sp.decode(tgt.squeeze().tolist())

print(translate("hello"))
