# AI Meeting Assistant

**General Assembly Data Science Bootcamp — Capstone Project**

An AI-powered meeting assistant that converts uploaded recordings into timestamped chapters and searchable transcripts, answers questions about individual meetings, and helps users draft follow-up emails.

## Problem & Motivation

Important decisions and action items can be difficult to find in long meeting recordings. This project aims to make meeting content easier to review and search without replaying the entire recording.

## Dataset & Sources

The application processes meeting videos uploaded by the user. These recordings form the project dataset; no external or public training dataset is included. Audio, transcripts, chapters, and embeddings are generated from each recording and stored locally under `data/`.


## Model Architecture & Pipeline

The project uses pre-trained models and API services; **no model is trained or fine-tuned** as part of this capstone.

1. **Audio extraction:** FFmpeg converts the uploaded video to mono, 16 kHz WAV audio.
2. **Audio chunking:** NumPy and FFmpeg split audio around silences (−30 dB, 0.3-second silence), targeting 4–20-second chunks with 0.2-second padding.
3. **Speech recognition:** Whisper Large-v3 transcribes audio chunks through the OpenRouter API.
4. **Transcript cleaning:** Python normalizes Unicode, removes Arabic diacritics, and flags empty or repetitive chunks as unclear speech.
5. **Chapter generation:** Nemotron 3 Ultra via OpenRouter creates timestamped chapter titles and summaries from transcript windows. Generated timestamps are checked against transcript timestamps.
6. **RAG indexing:** `RecursiveCharacterTextSplitter` creates 1,000-character chunks with 150-character overlap. BGE-M3 (`BAAI/bge-m3`) creates embeddings, which are stored with meeting and chapter metadata in ChromaDB.
7. **Question answering:** The selected meeting's relevant transcript chunks are retrieved from ChromaDB and sent with the question to the language model. The answer is instructed to use the transcript and cite retrieved sources.
8. **Email drafting:** The selected meeting transcript is used to generate an editable follow-up email. Users can review and send it through Resend.

## Evaluation & Baseline Comparison

ASR models were compared qualitatively on Arabic, English, and mixed Arabic/English recordings during development. The observations below are qualitative, not formal benchmark scores.

| ASR model | Arabic | English | Mixed Arabic/English |
|---|---|---|---|
| Whisper Large-v3 | Good; errors on names and terms | Very good | May translate or omit speech during language switching; occasional repetition |
| Qwen3-ASR 1.7B | Readable, with more errors in the tested samples | Very good | May translate Arabic into English |
| Jais-ASR | Strongest on the tested formal Arabic and dialect samples | Good | May render English words in Arabic script |
| Nemotron 3.5 ASR Streaming 0.6B | Captured Arabic in the tested samples | English output was garbled | Not reliable in the tested samples |

Whisper Large-v3 was selected as the project's ASR model because it provided a practical balance across the tested languages. No ground-truth corpus, Word Error Rate (WER), or Character Error Rate (CER) benchmark was established, so these results should be treated as development observations rather than a standardized evaluation.


## Repository Structure

```text
project/
├── app.py                  # Streamlit application and processing pipeline
├── cap_0_1.ipynb           # Development notebook / pipeline exploration
├── requirements.txt        # Python dependencies
├── .env                    # Local API keys (not committed)
├── data/                   # Generated files (not committed)
│   ├── videos/
│   ├── audio/
│   ├── pieces/
│   ├── transcripts/
│   └── chromadb/
└── README.md
```

## Limitations & Future Work

- Mixed Arabic/English speech can be mistranscribed, translated, or omitted; names and technical terms may also be inaccurate.
- The transcript does not currently include speaker diarization or speaker labels.
- Chapter summaries, retrieved answers, and generated emails can contain errors; users should review them before relying on or sending them.
- The current prototype is intended for local use and does not yet provide user authentication or production-grade multi-user data isolation.
- Future work could include quantitative ASR evaluation with a labelled test set, speaker diarization, improved multilingual transcription, stronger retrieval for multi-part questions, user accounts, and cloud deployment.
