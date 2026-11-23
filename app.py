import os
import re
import uuid
import subprocess
import json
import chromadb
import requests
import streamlit as st
from FlagEmbedding import BGEM3FlagModel
from openai import OpenAI
import wave
import numpy as np
from pathlib import Path
import unicodedata
import json
import time
import shutil
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Meeting Assistant", page_icon="📝")

@st.cache_resource
def get_embedding_model():
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

@st.cache_resource
def get_collection():
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    return db.get_or_create_collection(COLLECTION_NAME)

def get_meetings():

    collection = get_collection()

    data = collection.get(include=["metadatas"])

    meetings = {}

    for meta in data["metadatas"]:

        if not meta:
            continue

        meeting_id = meta.get("meeting_id")

        if meeting_id:
            meetings[meeting_id] = (
                meta.get("meeting_title")
                or f"Meeting {meeting_id[:8]}"
            )

    return meetings


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_bad(text):
    words = text.split()

    if not words:
        return True

    if re.search(r"(.)\1{9,}", text):
        return True

    top = max(set(words), key=words.count)

    return len(words) >= 8 and words.count(top) / len(words) > 0.5


def clean_transcript(chunks):
    out = []

    for c in chunks:
        text = clean_text(c["text"])
        bad = is_bad(text)

        out.append({
            **c,
            "text_clean": "[unclear speech]" if bad else text,
            "masked": bad
        })

    return out

def get_secret(name):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return None

PROMPT = """You are helping to organize a meeting transcript.

Below is part of a transcript. Each line starts with the time it begins, like [12:30].
Divide this part into 2 to 5 chapters, in time order.

For each chapter give:
- start: the time of the first line of the chapter, copied exactly from the transcript
- title: at most eight words
- summary: one sentence about what happens in the chapter

Rules:
- Use only what is written in the transcript. Never invent names, numbers, or events.
- Names may be spelled in different ways. If you are not sure who is speaking, say "a speaker".
- Lines marked [unclear speech] are gaps. Do not treat them as content.
- Write the titles and summaries in English.
- Do not add any name, title, or fact that is not written in the text.
- Each topic gets one chapter.
- Each chapter's start time must be the first line about that topic.

TRANSCRIPT:
"""

CHAPTER_SCHEMA = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"}
                },
                "required": ["start", "title", "summary"]
            }
        }
    },
    "required": ["chapters"]
}

def store_in_chromadb(rag_chunks, meeting_id, meeting_title):

    collection = get_collection()

    collection.upsert(
        ids=[
            f"{meeting_id}_chunk_{i}"
            for i in range(len(rag_chunks))
        ],

        documents=[
            chunk["text"]
            for chunk in rag_chunks
        ],

        embeddings=[
            chunk["embedding"]
            for chunk in rag_chunks
        ],

        metadatas=[
            {
                "meeting_id": meeting_id,
                "chapter": chunk["chapter"],
                "chapter_summary": chunk["chapter_summary"],
                "chapter_start": chunk["chapter_start"],
                "chapter_index": chunk["chapter_index"],
                "meeting_title": meeting_title or f"Meeting {meeting_id[:8]}"
            }
            for chunk in rag_chunks
        ]
    )

    return collection.count()


@st.cache_resource
def get_tokenizer():
    return AutoTokenizer.from_pretrained(
        "FacebookAI/xlm-roberta-base"
    )


def mmss(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def make_windows(chunks, max_tokens=3000):

    tokenizer = get_tokenizer()

    windows = []
    current = []
    count = 0

    for chunk in chunks:

        line = (
            f"[{mmss(chunk['start'])}] "
            f"{chunk['text_clean']}"
        )

        tokens = len(
            tokenizer.encode(
                line,
                add_special_tokens=False
            )
        )

        if current and count + tokens > max_tokens:
            windows.append("\n".join(current))
            current = []
            count = 0

        current.append(line)
        count += tokens

    if current:
        windows.append("\n".join(current))

    return windows


def generate_chapters(window_text):

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=get_secret("OPENROUTER"),
        max_retries=0
    )

    return client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",

        extra_body={
            "reasoning": {
                "effort": "minimal"
            }
        },

        messages=[
            {
                "role": "user",
                "content": PROMPT + window_text
            }
        ],

        temperature=0.2,
        max_tokens=10000,

        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "chapters",
                "strict": True,
                "schema": CHAPTER_SCHEMA
            }
        }
    )


def create_chapters(clean, meeting_id):

    windows = make_windows(clean)
    all_chapters = []

    transcript_dir = Path("data/transcripts")
    progress_path = transcript_dir / f"{meeting_id}_chapters_progress.json"
    final_path = transcript_dir / f"{meeting_id}_chapters.json"

    progress = st.progress(0)

    for i, window_text in enumerate(windows):

        response = generate_chapters(window_text)
        choice = response.choices[0]

        if choice.finish_reason == "length":
            raise ValueError("Chapter response was truncated.")

        if not choice.message.content:
            raise ValueError("Empty chapter response.")

        chapters = json.loads(
            choice.message.content
        )["chapters"]

        # Validate timestamps against the transcript
        valid_starts = {
            line.split("]")[0][1:]
            for line in window_text.splitlines()
            if line.startswith("[") and "]" in line
        }

        for chapter in chapters:

    # Remove square brackets from the timestamp
            chapter["start"] = chapter["start"].strip().strip("[]")

            if chapter["start"] not in valid_starts:
                raise ValueError(
                    f"Invalid chapter timestamp: {chapter['start']}"
                )

        all_chapters.extend(chapters)

        # Save progress
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(
                all_chapters,
                f,
                ensure_ascii=False,
                indent=2
            )

        progress.progress((i + 1) / len(windows))

        time.sleep(2)

    # Save final chapters
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(
            {"chapters": all_chapters},
            f,
            ensure_ascii=False,
            indent=2
        )

    return all_chapters

def to_seconds(timestamp):
    minutes, seconds = map(int, timestamp.split(":"))
    return minutes * 60 + seconds


def attach_chapters(clean, chapters):

    for i, chapter in enumerate(chapters):

        chapter_start = to_seconds(chapter["start"])

        if i + 1 < len(chapters):
            chapter_end = to_seconds(chapters[i + 1]["start"])
        else:
            chapter_end = float("inf")

        for chunk in clean:

            if chapter_start <= chunk["start"] < chapter_end:

                chunk["chapter"] = chapter["title"]
                chunk["chapter_summary"] = chapter["summary"]
                chunk["chapter_index"] = i
                chunk["chapter_start"] = chapter["start"]

    return clean

def create_rag_chunks(clean, meeting_id):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chapters_text = {}

    # Group transcript text by chapter
    for chunk in clean:

        index = chunk["chapter_index"]

        if index not in chapters_text:
            chapters_text[index] = {
                "chapter": chunk["chapter"],
                "chapter_summary": chunk["chapter_summary"],
                "start": chunk["chapter_start"],
                "text": ""
            }

        chapters_text[index]["text"] += chunk["text_clean"] + " "

    # Split each chapter into smaller pieces
    rag_chunks = []

    for index, chapter in chapters_text.items():

        pieces = splitter.split_text(chapter["text"])

        for piece in pieces:
            rag_chunks.append({
                "text": piece,
                "chapter": chapter["chapter"],
                "chapter_summary": chapter["chapter_summary"],
                "chapter_start": chapter["start"],
                "chapter_index": index
            })

    # Save RAG chunks
    rag_dir = Path("data/transcripts")
    rag_dir.mkdir(parents=True, exist_ok=True)

    rag_path = rag_dir / f"{meeting_id}_rag_chunks.json"

    with open(rag_path, "w", encoding="utf-8") as f:
        json.dump(rag_chunks, f, ensure_ascii=False, indent=2)

    return rag_chunks

def create_embeddings(rag_chunks, meeting_id):

    if not rag_chunks:
        raise ValueError("No RAG chunks were created.")

    model = get_embedding_model()

    texts = [chunk["text"] for chunk in rag_chunks]

    embeddings = model.encode(
        texts,
        batch_size=8,
        max_length=8192
    )

    for i, chunk in enumerate(rag_chunks):
        chunk["embedding"] = embeddings["dense_vecs"][i].tolist()

    # Save embedded chunks
    output_path = (
        Path("data/transcripts")
        / f"{meeting_id}_rag_chunks_embedded.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            rag_chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    return rag_chunks

def transcribe_chunks(chunks, meeting_id):

    api_key = get_secret("OPENROUTER")

    if not api_key:
        raise ValueError("OPENROUTER API key is missing.")

    stt_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_retries=2,
        timeout=120.0
    )

    out = []

    transcript_dir = Path("data/transcripts")
    transcript_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = transcript_dir / f"{meeting_id}.json"

    progress = st.progress(0)

    for i, chunk in enumerate(chunks):

        with open(chunk["file"], "rb") as audio_file:

            r = stt_client.audio.transcriptions.create(
                model="openai/whisper-large-v3",
                file=audio_file,
                response_format="json"
            )

        out.append({
            "chunk_index": i,
            "start": chunk["start"],
            "end": chunk["end"],
            "text": r.text.strip()
        })

        # Save progress after every chunk
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        progress.progress(
            (i + 1) / len(chunks)
        )

    return transcript_path

def split_audio(audio_path, meeting_id):

    pieces_dir = Path("data/pieces") / meeting_id
    pieces_dir.mkdir(parents=True, exist_ok=True)

    # Get audio duration
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ],
        capture_output=True,
        text=True,
        check=True
    )

    duration = float(result.stdout.strip())

    # Load audio
    with wave.open(str(audio_path), "rb") as w:
        sr = w.getframerate()
        samples = np.frombuffer(
            w.readframes(w.getnframes()),
            dtype=np.int16
        ).astype(np.float32)

    WIN = int(0.1 * sr)
    n = len(samples) // WIN

    if n == 0:
        raise ValueError("Audio file is empty or too short.")

    rms = np.sqrt(
        (samples[:n * WIN].reshape(n, WIN) ** 2).mean(axis=1)
    )

    def quietest_cut(lo, hi):
        a = int(lo / 0.1)
        b = min(int(hi / 0.1), n)

        if a >= b:
            return lo

        i = a + int(np.argmin(rms[a:b]))
        return (i + 0.5) * 0.1

    # Detect silence
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(audio_path),
            "-af", "silencedetect=noise=-30dB:d=0.3",
            "-f", "null", "-"
        ],
        capture_output=True,
        text=True
    )

    starts = re.findall(
        r"silence_start:\s*([\d.]+)",
        result.stderr
    )

    ends = re.findall(
        r"silence_end:\s*([\d.]+)",
        result.stderr
    )

    silences = [
        (float(s), float(e))
        for s, e in zip(starts, ends)
    ]

    MIN_LEN = 4
    TARGET_LEN = 12
    MAX_LEN = 20
    PADDING = 0.2

    boundaries = [0.0]
    current = 0.0

    while duration - current > MAX_LEN:

        candidates = [
            (s + e) / 2
            for s, e in silences
            if current + MIN_LEN <= (s + e) / 2 <= current + MAX_LEN
        ]

        if candidates:
            cut = min(
                candidates,
                key=lambda x: abs(x - (current + TARGET_LEN))
            )
        else:
            cut = quietest_cut(
                current + MIN_LEN,
                current + MAX_LEN
            )

        boundaries.append(cut)
        current = cut

    boundaries.append(duration)

    # Export chunks
    chunks = []

    for i, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):

        export_start = (
            max(0, start - PADDING) if i > 0 else start
        )

        export_end = (
            min(duration, end + PADDING)
            if i < len(boundaries) - 2
            else end
        )

        output_file = pieces_dir / f"p_{i:04d}.wav"

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(export_start),
                "-i", str(audio_path),
                "-t", str(export_end - export_start),
                "-ac", "1",
                "-ar", "16000",
                str(output_file)
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        chunks.append({
            "file": str(output_file),
            "start": export_start,
            "end": export_end
        })

    return chunks

# Basic settings
CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    "data/chromadb"
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "distractible_transcript")
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

st.title("Meeting Assistant")


video_dir = Path("data/videos")
audio_dir = Path("data/audio")

video_dir.mkdir(parents=True, exist_ok=True)
audio_dir.mkdir(parents=True, exist_ok=True)

meeting_title = st.text_input("Meeting title")

uploaded_video = st.file_uploader(
    "Upload meeting video",
    type=["mp4", "mov", "mkv", "webm"]
)

if uploaded_video:
    st.video(uploaded_video)

    if st.button("Process meeting"):

        meeting_id = str(uuid.uuid4())

        
        video_path = None
        audio_path = None
        pieces_dir = Path("data/pieces") / meeting_id

        try:
            with st.spinner("Processing video..."):

                
                video_path = video_dir / (
                    f"{meeting_id}_{Path(uploaded_video.name).name}"
                )

                with open(video_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())

                
                audio_path = audio_dir / f"{meeting_id}.wav"

                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", str(video_path),
                        "-vn",
                        "-ac", "1",
                        "-ar", "16000",
                        str(audio_path)
                    ],
                    check=True,
                    capture_output=True
                )

                
                chunks = split_audio(audio_path, meeting_id)

                
                transcript_path = transcribe_chunks(chunks, meeting_id)

                with open(transcript_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)

                
                clean = clean_transcript(raw)

                
                chapters = create_chapters(clean, meeting_id)
                clean = attach_chapters(clean, chapters)

                
                rag_chunks = create_rag_chunks(clean, meeting_id)

                
                rag_chunks = create_embeddings(rag_chunks, meeting_id)

                
                count = store_in_chromadb(
                    rag_chunks,
                    meeting_id,
                    meeting_title
                )

                
                clean_path = (
                    Path("data/transcripts") /
                    f"{meeting_id}_clean.json"
                )

                with open(clean_path, "w", encoding="utf-8") as f:
                    json.dump(clean, f, ensure_ascii=False, indent=2)

            st.success("Meeting processed successfully!")
            st.write(f"Created {len(chunks)} audio chunks.")
            st.write(f"Created {len(rag_chunks)} RAG chunks.")
            st.write(f"Total chunks in ChromaDB: {count}")

            st.subheader("Generated chapters")

            for chapter in chapters:
                st.write(
                    f"{chapter['start']} | {chapter['title']}"
                )
                st.caption(chapter["summary"])

        except Exception as error:
            st.error(f"Processing failed: {error}")

        finally:
            
            if video_path and video_path.exists():
                video_path.unlink()

            
            if audio_path and audio_path.exists():
                audio_path.unlink()

            
            shutil.rmtree(pieces_dir, ignore_errors=True)


def ask_openrouter(prompt, system_prompt):
    api_key = get_secret("OPENROUTER") or get_secret("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Add your OPENROUTER key to the environment or Streamlit secrets.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=180,
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1600,
    )
    return response.choices[0].message.content or "No answer was returned."


def get_transcript(meeting_id):

    transcript_path = (
        Path("data/transcripts") /
        f"{meeting_id}_clean.json"
    )

    if not transcript_path.exists():
        return ""

    with open(transcript_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    chunks.sort(key=lambda x: x["start"])

    return "\n\n".join(
        f"[{mmss(chunk['start'])}] {chunk['text_clean']}"
        for chunk in chunks
        if chunk.get("text_clean")
    )

st.divider()
st.header("Your meetings")

meetings = get_meetings()

if meetings:

    selected_meeting_id = st.selectbox(
        "Select a meeting",
        options=list(meetings.keys()),
        format_func=lambda mid: meetings[mid]
    )

else:
    selected_meeting_id = None
    st.info("No meetings available. Upload and process a meeting first.")

selected_meeting_title = (
    meetings[selected_meeting_id]
    if selected_meeting_id
    else ""
)


ask_tab, transcript_tab, email_tab = st.tabs([
    "Ask the transcript",
    "View transcript",
    "Post-meeting email"
])

with ask_tab:
    question = st.text_input("Ask a question about the meeting")

    if st.button("Ask", key="ask"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                collection = get_collection()
                if not selected_meeting_id:
                    st.warning("Select a meeting first.")
                else:
                    meeting_data = collection.get(
                        where={"meeting_id": selected_meeting_id},
                        include=["metadatas"]
                    )

                    if not meeting_data["ids"]:
                        st.warning("No chunks found for this meeting.")
                    else:
                        with st.spinner("Searching the transcript..."):
                            vector = get_embedding_model().encode(
                                [question], batch_size=1
                            )["dense_vecs"][0].tolist()

                            found = collection.query(
                                query_embeddings=[vector],
                                where={"meeting_id": selected_meeting_id},
                                n_results=min(5, len(meeting_data["ids"])),
                            )

                            passages = []
                            for i, (doc, meta) in enumerate(zip(found["documents"][0], found["metadatas"][0]), 1):
                                meta = meta or {}
                                passages.append(
                                    f"[Source {i}] Chapter: {meta.get('chapter', 'Unknown')} "
                                    f"(starts {meta.get('chapter_start', 'unknown')})\n{doc}"
                                )

                            context = "\n\n".join(passages)
                            answer = ask_openrouter(
                                f"Transcript passages:\n{context}\n\nQuestion: {question}",
                                "Answer only from the transcript passages. Do not invent facts. "
                                "Say when the transcript does not provide enough information. "
                                "Cite sources as [Source 1], [Source 2], etc.",
                            )

                        st.subheader("Answer")
                        st.write(answer)
                        with st.expander("See retrieved transcript passages"):
                            for passage in passages:
                                st.write(passage)
                                st.divider()
            except Exception as error:
                st.error(str(error))

with transcript_tab:

    if selected_meeting_id:

        transcript = get_transcript(selected_meeting_id)

        if transcript:
            st.text_area(
                "Meeting transcript",
                value=transcript,
                height=500,
                disabled=True
            )

            st.download_button(
                "Download transcript",
                data=transcript,
                file_name=f"{selected_meeting_id}_transcript.txt",
                mime="text/plain"
            )

        else:
            st.info("No transcript found.")

    else:
        st.info("Select a meeting first.")

with email_tab:

    if not selected_meeting_id:
        st.info("Select a meeting first.")

    else:
        st.write(
            f"Generate a follow-up email for: {selected_meeting_title}"
        )

        subject_key = f"email_subject_{selected_meeting_id}"
        body_key = f"email_body_{selected_meeting_id}"
        sent_key = f"email_sent_{selected_meeting_id}"

        recipient_name = st.text_input(
            "Recipient name (optional)",
            key=f"recipient_name_{selected_meeting_id}"
        )

        recipient_email = st.text_input(
            "Send to email address",
            key=f"recipient_email_{selected_meeting_id}"
        )

        if st.button(
            "Generate email",
            key=f"generate_email_{selected_meeting_id}"
        ):

            try:
                transcript = get_transcript(selected_meeting_id)

                if not transcript.strip():
                    st.warning("No transcript was found.")
                else:
                    with st.spinner("Writing the email..."):

                        prompt = f"""
You are an experienced executive assistant. Your task is to write a clear, professional, well-structured follow-up email based on a meeting transcript.

MEETING DETAILS:
Meeting title: {selected_meeting_title}
Recipient: {recipient_name or "Use a professional greeting"}

REQUIREMENTS:

1. Write a natural, professional email that is ready to send.
2. Use a friendly but professional tone.
3. Be concise and avoid unnecessary repetition.
4. Use only information explicitly supported by the transcript.
5. Never invent decisions, agreements, action items, names, responsibilities, or deadlines.
6. Do not write a transcript or describe the conversation minute by minute.
7. Focus on the important outcomes and information the recipient needs.

EMAIL STRUCTURE:

Subject:
Write a short, relevant subject line that reflects the meeting.

Greeting:
Address the recipient appropriately.

Opening:
Start with a brief thank-you and explain the purpose of the follow-up.

Meeting Summary:
Summarize the main topics and important outcomes in a short paragraph.

Key Points:
Use bullet points to highlight the most important discussion points. Avoid repeating information already covered.

Decisions and Agreements:
Include only decisions that were clearly made or agreed upon during the meeting. Omit this section if none were identified.

Action Items / Next Steps:
List confirmed tasks using this format:
- Task:
- Responsible person: Only if explicitly identified.
- Deadline: Only if explicitly mentioned.

Closing:
End with a natural professional sentence inviting corrections or further discussion, followed by an appropriate sign-off.

OUTPUT RULES:
- Put "Subject: ..." on the first line.
- Use plain text, not HTML.
- Use clear paragraph spacing.
- Use bullet points only where they improve readability.
- Do not force every section into the email. Omit sections that have no relevant information.
- Do not include explanations, notes, or commentary outside the email.
- Aim for approximately 150–250 words, depending on the amount of useful information in the transcript.

TRANSCRIPT:
{transcript[:60000]}
"""

                        draft = ask_openrouter(
                            prompt,
                            "You write accurate meeting follow-up emails. Never invent details."
                        )

                    first_line, separator, body = draft.partition("\n")

                    if first_line.lower().startswith("subject:"):
                        st.session_state[subject_key] = (
                            first_line.split(":", 1)[1].strip()
                        )
                        st.session_state[body_key] = (
                            body.strip() if separator else ""
                        )
                    else:
                        st.session_state[subject_key] = (
                            f"Meeting summary: {selected_meeting_title}"
                        )
                        st.session_state[body_key] = draft

                    st.session_state[sent_key] = False

                    st.success("Email draft created!")

            except Exception as error:
                st.error(str(error))

        
        if body_key in st.session_state:

            st.subheader("Review email")

            subject = st.text_input(
                "Subject",
                key=subject_key
            )

            body = st.text_area(
                "Email body",
                key=body_key,
                height=300
            )

            valid_email = re.fullmatch(
                r"[^\s@]+@[^\s@]+\.[^\s@]+",
                recipient_email.strip()
            )

            if st.button(
                "Send email",
                key=f"send_email_{selected_meeting_id}",
                disabled=(
                    not valid_email
                    or st.session_state.get(sent_key, False)
                )
            ):

                try:
                    resend_key = get_secret("RESEND_API_KEY")
                    from_email = get_secret("RESEND_FROM_EMAIL")

                    if not resend_key or not from_email:
                        raise ValueError(
                            "Set RESEND_API_KEY and RESEND_FROM_EMAIL first."
                        )

                    result = requests.post(
                        "https://api.resend.com/emails",
                        headers={
                            "Authorization": f"Bearer {resend_key}",
                            "Content-Type": "application/json",
                            "Idempotency-Key": str(uuid.uuid4())
                        },
                        json={
                            "from": from_email,
                            "to": [recipient_email.strip()],
                            "subject": subject,
                            "text": body
                        },
                        timeout=30
                    )

                    if result.ok:
                        st.session_state[sent_key] = True
                        st.success("Email sent successfully!")
                    else:
                        st.error(f"Resend error: {result.text}")

                except Exception as error:
                    st.error(str(error))

