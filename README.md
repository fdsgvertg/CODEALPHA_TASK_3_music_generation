# TASK 3: Music Generation with AI

## Features
- Loads classical data from music21 Bach corpus
- Preprocesses into pitch + duration sequences
- LSTM model (PyTorch) for next-note prediction
- Training loop with validation
- Generates new music sequences
- Exports generated music as MIDI file

## Requirements
```bash
pip install music21 torch numpy
```

## How to run
```bash
python music_generation_ai.py
```

- Training takes a few minutes (15 epochs on CPU).
- Output: `generated_music.mid` and `models/music_lstm.pt`

## Tips for better results
- Increase `max_works` (e.g. 100)
- Train more epochs (30–80)
- Adjust temperature (0.7–1.1) when generating
- Play the MIDI in any DAW, VLC, or online MIDI player
