## zadanie
Encode and decode the **same** speech clips under several settings:
- different bitrates or number of codebooks,
- clean speech,
- noisy speech,
- expressive speech,
- long utterances,
- out-of-domain speakers.

Measure:
- ASR WER before and after reconstruction,
- speaker-embedding similarity,
- mel-spectrogram distance,
- file size or token rate,
- encode/decode speed,
- human ratings for artifacts.

Recommend codec settings for future speech-token training - lowest bitrate that keeps words intelligible, preserves speaker identity, and avoids obvious artifacts.

## modele
- whisper large v3 turbo 0.8b
- qwen asr 0.6b
- parakeet-tdt-v2 0.6b
- canary-flash 0.18b
- owsm-ctc-v3.2_ft 1b
- scribe v2 elevenlabs
kazdy model sprawdza sie do czego innego, jest troche inny caly czas

## dane
- wybrane wczesniej datasety
- rozne kodeki (30 wariantow)
kazdy kodek powinien miec swoje podzbiory:
- clean speech
- noisy speech
- long utterances

## metryki
z zadania:
- wer
- speaker-embedding similarity
- mel-spectrogram distance
- file size or token rate
- encode/decode speed
- human ratings for artifacts
propozycje:
- RTFx - duration/processing time - czasowa metryka
- CER - character error rate liczony podobnie jak wer ale na poziomie znakow
- MER - missed entity rate (wymaga konfiguracji) - chodzi o jakies liczby lub maile lub inne specyficzne nazwy wlasne
- [SeMaScore](https://arxiv.org/pdf/2401.07506) - improve bertscore o 41x szybkosci, chodzi o znaczenie pelnych zdan zazwyczaj czy sie zgadza

## 