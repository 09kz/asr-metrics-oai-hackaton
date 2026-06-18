# ASR Metrics

A comprehensive toolkit for evaluating Automatic Speech Recognition (ASR) systems. Computes text-level, audio-level, and performance metrics in a single batch call.

## Metrics

### Text Metrics
| Metric | Description |
|--------|-------------|
| **WER** | Word Error Rate — fraction of word-level edits (insertions, deletions, substitutions) |
| **CER** | Character Error Rate — same as WER but at the character level |
| **MER** | Match Error Rate — ratio of matching errors to total alignment length |
| **SeMaScore** | Semantic Match Score — edit-distance-aligned cosine similarity weighted by segment importance (uses RoBERTa embeddings) |

### Audio Metrics
| Metric | Description |
|--------|-------------|
| **Speaker Similarity** | Cosine similarity of ECAPA-TDNN speaker embeddings between reference and predicted audio |
| **Mel Distance** | Mean squared error between mel spectrograms (with DTW fallback for different lengths) |

### Performance Metrics
| Metric | Description |
|--------|-------------|
| **RTFX** | Real-Time Factor — ratio of decode time to audio duration |
| **Token Rate** | Tokens produced per second of decode time |
| **Encode / Decode Time** | Raw timing of the ASR encode and decode stages |
| **File Size** | Output audio file size in bytes |
| **Audio Duration** | Output audio duration in seconds |

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch
- [jiwer](https://github.com/jitsi/jiwer) — WER / CER / MER
- [SpeechBrain](https://github.com/speechbrain/speechbrain) — speaker embeddings (ECAPA-TDNN)
- [librosa](https://librosa.org/) + [soundfile](https://pysoundfile.readthedocs.io/) — mel spectrogram & audio I/O
- [transformers](https://huggingface.co/docs/transformers/) — RoBERTa model for SeMaScore

> **Windows note:** SpeechBrain may fail to create symlinks without admin privileges. The toolkit includes an automatic fallback that copies files instead.

## Usage

```python
from metrics import eval_all_metrics, preload_models

# Optional: preload heavy models once
preload_models()

# Evaluate a batch
aggregates, individual = eval_all_metrics(
    ytrue_texts=["the cat sat on the mat"],
    ypred_texts=["the cat set on a mat"],
    ytrue_audio_paths=["samples/sample1.flac"],
    ypred_audio_paths=["samples/sample2.flac"],
    decode_times=[0.45],
    encode_times=[0.12],
    log=True,
)

print(aggregates)
# {'wer': 0.333, 'cer': 0.15, 'mer': 0.333, 'semascore': 0.92, ...}

print(individual[0])
# Per-sample breakdown with all metrics
```

### Using SeMaScore Standalone

```python
from sema_score_generator import generate_sema_score

result = generate_sema_score(
    "the quick brown fox",
    "the quik brown fox",
    verbose=True,  # enable debug output
)
score = result[0]
```

## Project Structure

```
├── metrics.py                 # Main evaluation API (eval_all_metrics)
├── sema_score_generator.py    # SeMaScore implementation
├── requirements.txt           # Python dependencies
├── samples/                   # Sample audio files for testing
│   ├── sample1.flac
│   ├── sample2.flac
│   └── sample3.flac
└── showcase.ipynb             # Jupyter notebook demo
```

## Credits

- **SeMaScore** implementation is based on [zenlab-edgeASR/SeMaScore](https://github.com/zenlab-edgeASR/SeMaScore/tree/main/codes).
- **Speaker similarity** uses the [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) model from SpeechBrain.

## License

This project was created during the OAI Camp hackathon.
