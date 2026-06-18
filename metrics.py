import os
import shutil
import time
import contextlib
import io
import jiwer
import torch
import librosa
import numpy as np
import soundfile as sf

# Monkey-patch os.symlink on Windows to fallback to copying to prevent SpeechBrain [WinError 1314]
if os.name == 'nt':
    _orig_symlink = os.symlink
    def safe_symlink(src, dst, *args, **kwargs):
        try:
            _orig_symlink(src, dst, *args, **kwargs)
        except OSError as e:
            if e.winerror == 1314:  # Privilege not held
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            else:
                raise
    os.symlink = safe_symlink

speaker_model = None

def get_speaker_model():
    global speaker_model
    if speaker_model is None:
        from speechbrain.inference.speaker import EncoderClassifier
        speaker_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir")
        import sys
        # Remove broken lazy modules injected by speechbrain that crash librosa's inspect.stack()
        for k in list(sys.modules.keys()):
            if 'speechbrain.integrations' in k:
                del sys.modules[k]
    return speaker_model

def preload_models():
    """Wczytuje modele (SpeechBrain i SeMaScore) do pamięci RAM/VRAM z wyprzedzeniem."""
    print("Preloading SpeechBrain model...")
    get_speaker_model()
    print("Preloading SeMaScore model (roberta-base)...")
    import sema_score_generator
    print("Models preloaded successfully!")

def compute_wer_cer_mer(ytrue_texts, ypred_texts):
    results = {'wer': [], 'cer': [], 'mer': []}
    for yt, yp in zip(ytrue_texts, ypred_texts):
        results['wer'].append(jiwer.wer(yt, yp))
        results['cer'].append(jiwer.cer(yt, yp))
        results['mer'].append(jiwer.mer(yt, yp))
    return results

def compute_speaker_similarity(ytrue_audio_paths, ypred_audio_paths, log=False):
    model = get_speaker_model()
    
    if log: print(f"Loading {len(ytrue_audio_paths)} audio pairs for speaker similarity...")
    signals1 = [model.load_audio(p) for p in ytrue_audio_paths]
    signals2 = [model.load_audio(p) for p in ypred_audio_paths]
    
    from torch.nn.utils.rnn import pad_sequence
    wavs1 = pad_sequence(signals1, batch_first=True)
    lens1 = torch.tensor([len(s)/wavs1.shape[1] for s in signals1])
    emb1 = model.encode_batch(wavs1, wav_lens=lens1).squeeze()
    if emb1.dim() == 1: emb1 = emb1.unsqueeze(0)
    
    wavs2 = pad_sequence(signals2, batch_first=True)
    lens2 = torch.tensor([len(s)/wavs2.shape[1] for s in signals2])
    emb2 = model.encode_batch(wavs2, wav_lens=lens2).squeeze()
    if emb2.dim() == 1: emb2 = emb2.unsqueeze(0)
    
    sims = torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1)
    return sims.tolist()

def compute_mel_distance(ytrue_audio_paths, ypred_audio_paths, log=False):
    distances = []
    if log: print(f"Computing mel distance for {len(ytrue_audio_paths)} pairs...")
    for yt, yp in zip(ytrue_audio_paths, ypred_audio_paths):
        y1, sr1 = sf.read(yt)
        y2, sr2 = sf.read(yp)
        mel1_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y1, sr=sr1), ref=np.max)
        mel2_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y2, sr=sr2), ref=np.max)
        
        if mel1_db.shape == mel2_db.shape:
            distances.append(float(np.mean((mel1_db - mel2_db)**2)))
        else:
            D, wp = librosa.sequence.dtw(mel1_db, mel2_db)
            distances.append(float(D[-1, -1]))
    return distances

def compute_semascore(ytrue_texts, ypred_texts, log=False):
    from sema_score_generator import generate_sema_score
    scores = []
    if log: print(f"Computing SeMaScore for {len(ytrue_texts)} pairs...")
    for yt, yp in zip(ytrue_texts, ypred_texts):
        if log:
            res = generate_sema_score(yt, yp)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                res = generate_sema_score(yt, yp)
        scores.append(res[0])
    return scores

def eval_all_metrics(
    ytrue_texts=None, 
    ypred_texts=None, 
    ytrue_audio_paths=None, 
    ypred_audio_paths=None, 
    decode_times=None, 
    encode_times=None,
    log=False
):
    """
    Evaluates ASR metrics for a batch of predictions.
    
    Args:
        ytrue_texts (list[str]): Ground truth transcriptions.
        ypred_texts (list[str]): Predicted transcriptions.
        ytrue_audio_paths (list[str]): Ground truth audio paths.
        ypred_audio_paths (list[str]): Predicted audio paths.
        decode_times (list[float]): ASR decode times for RTFX and Token Rate calculation.
        encode_times (list[float]): ASR encode times.
        log (bool): Enable verbose logging.
        
    Returns:
        tuple (dict, dict): Aggregate metrics and individual metrics per pair.
    """
    N = 0
    if ytrue_texts is not None: N = len(ytrue_texts)
    elif ytrue_audio_paths is not None: N = len(ytrue_audio_paths)
    
    if log: print(f"Evaluating metrics for {N} samples in batch...")

    individual = {i: {} for i in range(N)}
    aggregates = {}

    if ytrue_texts and ypred_texts:
        wcm = compute_wer_cer_mer(ytrue_texts, ypred_texts)
        try:
            sema = compute_semascore(ytrue_texts, ypred_texts, log=log)
        except Exception as e:
            sema = [f"Error: {e}"] * N

        for i in range(N):
            individual[i]['wer'] = wcm['wer'][i]
            individual[i]['cer'] = wcm['cer'][i]
            individual[i]['mer'] = wcm['mer'][i]
            individual[i]['semascore'] = sema[i]

        aggregates['wer'] = sum(wcm['wer'])/N if N > 0 else 0
        aggregates['cer'] = sum(wcm['cer'])/N if N > 0 else 0
        aggregates['mer'] = sum(wcm['mer'])/N if N > 0 else 0
        if isinstance(sema[0], float):
            aggregates['semascore'] = sum(sema)/N if N > 0 else 0
        else:
            aggregates['semascore'] = sema[0]

    if ytrue_audio_paths and ypred_audio_paths:
        try:
            spk = compute_speaker_similarity(ytrue_audio_paths, ypred_audio_paths, log=log)
        except Exception as e:
            spk = [f"Error: {e}"] * N
            
        try:
            mel = compute_mel_distance(ytrue_audio_paths, ypred_audio_paths, log=log)
        except Exception as e:
            mel = [f"Error: {e}"] * N

        for i in range(N):
            individual[i]['speaker_similarity'] = spk[i]
            individual[i]['mel_distance'] = mel[i]

        if isinstance(spk[0], float):
            aggregates['speaker_similarity'] = sum(spk)/N if N > 0 else 0
        else:
            aggregates['speaker_similarity'] = spk[0]
            
        if isinstance(mel[0], float):
            aggregates['mel_distance'] = sum(mel)/N if N > 0 else 0
        else:
            aggregates['mel_distance'] = mel[0]

    if ypred_audio_paths:
        file_sizes = [os.path.getsize(p) for p in ypred_audio_paths]
        audio_durations = [sf.info(p).duration for p in ypred_audio_paths]
        for i in range(N):
            individual[i]['file_size'] = file_sizes[i]
            individual[i]['audio_duration'] = audio_durations[i]
        aggregates['file_size'] = sum(file_sizes)/N if N > 0 else 0
        aggregates['audio_duration'] = sum(audio_durations)/N if N > 0 else 0
    else:
        audio_durations = [None] * N

    if ypred_texts:
        token_counts = [len(t.split()) for t in ypred_texts]
        for i in range(N):
            individual[i]['token_count'] = token_counts[i]
        aggregates['token_count'] = sum(token_counts)/N if N > 0 else 0
    else:
        token_counts = [None] * N

    for i in range(N):
        dt = decode_times[i] if decode_times else None
        et = encode_times[i] if encode_times else None
        tc = token_counts[i]
        ad = audio_durations[i]
        
        individual[i]['encode_time'] = et
        individual[i]['decode_time'] = dt
        individual[i]['token_rate'] = (tc / dt) if tc is not None and dt else None
        individual[i]['rtfx'] = (dt / ad) if dt is not None and ad else None

    # Aggregates for these
    if decode_times:
        aggregates['decode_time'] = sum(decode_times)/N if N > 0 else 0
    if encode_times:
        aggregates['encode_time'] = sum(encode_times)/N if N > 0 else 0
        
    valid_tr = [ind['token_rate'] for ind in individual.values() if ind['token_rate'] is not None]
    aggregates['token_rate'] = sum(valid_tr)/len(valid_tr) if valid_tr else None

    valid_rtfx = [ind['rtfx'] for ind in individual.values() if ind['rtfx'] is not None]
    aggregates['rtfx'] = sum(valid_rtfx)/len(valid_rtfx) if valid_rtfx else None

    return aggregates, individual

