# music_generation_ai.py
# TASK 3: Music Generation with AI
# Uses music21 corpus (Bach) + PyTorch LSTM

import os
import random
import numpy as np
from collections import Counter
from music21 import corpus, note, chord, stream
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------
# 1. Collect / Load MIDI-like data (Bach chorales)
# -------------------------------------------------
def load_bach_notes(max_works=30):
    """Load notes (pitch + duration) from a subset of Bach chorales."""
    works = corpus.getComposer('bach')[:max_works]
    all_notes = []
    
    for path in works:
        try:
            score = corpus.parse(path)
            # Flatten and extract notes/chords
            for el in score.flatten().notes:
                if isinstance(el, note.Note):
                    # Encode as "pitch_duration" string for simplicity
                    p = el.pitch.midi
                    d = round(float(el.duration.quarterLength), 2)
                    all_notes.append(f"{p}_{d}")
                elif isinstance(el, chord.Chord):
                    # Take root of chord for simplicity
                    p = el.root().midi if el.root() else el.pitches[0].midi
                    d = round(float(el.duration.quarterLength), 2)
                    all_notes.append(f"{p}_{d}")
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
    
    print(f"Loaded {len(all_notes)} note events from {len(works)} works")
    return all_notes

notes = load_bach_notes(max_works=40)  # increase for better results

# -------------------------------------------------
# 2. Preprocess into sequences
# -------------------------------------------------
# Build vocabulary
unique_notes = sorted(list(set(notes)))
note_to_int = {n: i for i, n in enumerate(unique_notes)}
int_to_note = {i: n for n, i in note_to_int.items()}
vocab_size = len(unique_notes)
print(f"Vocabulary size: {vocab_size}")

# Create input sequences (window of SEQ_LEN notes → next note)
SEQ_LEN = 32
sequences = []
targets = []

for i in range(0, len(notes) - SEQ_LEN):
    seq = notes[i:i + SEQ_LEN]
    target = notes[i + SEQ_LEN]
    sequences.append([note_to_int[n] for n in seq])
    targets.append(note_to_int[target])

X = np.array(sequences, dtype=np.int64)
y = np.array(targets, dtype=np.int64)
print(f"Training sequences: {len(X)}")

# Train / val split
split = int(0.9 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

class NoteDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(NoteDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(NoteDataset(X_val, y_val), batch_size=64)

# -------------------------------------------------
# 3. LSTM Model
# -------------------------------------------------
class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)               # (B, T, E)
        out, hidden = self.lstm(x, hidden)  # (B, T, H)
        out = self.dropout(out[:, -1, :])   # last time step
        logits = self.fc(out)               # (B, vocab)
        return logits, hidden

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicLSTM(vocab_size).to(device)
print(f"Model on {device} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

# -------------------------------------------------
# 4. Train
# -------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits, _ = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits, _ = model(Xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()
    return total_loss / len(loader)

EPOCHS = 15   # increase to 30–50 for better musicality
for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader)
    val_loss   = eval_epoch(model, val_loader)
    scheduler.step()
    print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

# Save checkpoint
os.makedirs("models", exist_ok=True)
torch.save({
    "model_state": model.state_dict(),
    "note_to_int": note_to_int,
    "int_to_note": int_to_note,
    "vocab_size": vocab_size,
    "seq_len": SEQ_LEN,
}, "models/music_lstm.pt")
print("Model saved.")

# -------------------------------------------------
# 5. Generate new sequences → MIDI
# -------------------------------------------------
def generate(model, seed_seq, length=100, temperature=1.0):
    model.eval()
    generated = list(seed_seq)
    hidden = None
    input_seq = torch.tensor([seed_seq], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_seq, hidden)
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
            next_idx = np.random.choice(len(probs), p=probs)
            generated.append(next_idx)
            # slide window
            input_seq = torch.tensor([[next_idx]], dtype=torch.long).to(device)
    return generated

# Pick a random seed from training data
seed_idx = random.randint(0, len(X_train) - 1)
seed = X_train[seed_idx].tolist()
generated_ints = generate(model, seed, length=120, temperature=0.9)

# Decode to pitch + duration
generated_notes = [int_to_note[i] for i in generated_ints]

# Build music21 stream
s = stream.Stream()
s.append(note.Rest(quarterLength=0.5))  # small lead-in

for token in generated_notes:
    try:
        p_str, d_str = token.split("_")
        p = int(p_str)
        d = float(d_str)
        n = note.Note()
        n.pitch.midi = p
        n.duration.quarterLength = max(0.25, d)  # avoid zero
        s.append(n)
    except:
        continue

# Write MIDI
out_path = "generated_music.mid"
s.write("midi", fp=out_path)
print(f"Generated MIDI saved to: {out_path}")
print(f"Generated {len(s.notes)} notes")
