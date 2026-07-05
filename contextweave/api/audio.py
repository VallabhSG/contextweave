"""Voice ingestion: transcribe audio uploads into conversation memories."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from contextweave.api.deps import get_workspace
from contextweave.api.pipeline import IngestResponse, process_events
from contextweave.api.rate_limit import INGEST_LIMIT, limiter
from contextweave.config import settings
from contextweave.ingestion.text_adapter import TextAdapter
from contextweave.schemas import SourceType
from contextweave.workspaces import Workspace

logger = logging.getLogger(__name__)

router = APIRouter()

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".mp4", ".wav", ".webm", ".ogg", ".flac", ".mpeg", ".mpga"}
MAX_AUDIO_BYTES = 15 * 1024 * 1024  # Groq caps Whisper uploads at 25 MB


class AudioIngestResponse(IngestResponse):
    transcript: str = ""


def _transcribe(filename: str, data: bytes) -> str:
    """Speech-to-text via Groq Whisper."""
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    result = client.audio.transcriptions.create(
        file=(filename, data),
        model=settings.transcription_model,
    )
    return (result.text or "").strip()


@router.post("/ingest/audio", response_model=AudioIngestResponse)
@limiter.limit(INGEST_LIMIT)
async def ingest_audio(
    request: Request, file: UploadFile = File(...), ws: Workspace = Depends(get_workspace)
):
    """Transcribe a voice recording and ingest it as a conversation memory."""
    filename = file.filename or "recording.m4a"
    suffix = Path(filename).suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise HTTPException(400, f"Unsupported audio format: {suffix or '(none)'}")

    data = await file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, f"Audio too large (max {MAX_AUDIO_BYTES // (1024 * 1024)} MB)")
    if not data:
        raise HTTPException(400, "Empty audio file")

    if not settings.groq_api_key:
        raise HTTPException(
            503,
            "Transcription needs CW_GROQ_API_KEY on the server — "
            "get a free key at console.groq.com",
        )

    try:
        transcript = await run_in_threadpool(_transcribe, filename, data)
    except Exception as e:
        logger.error("Transcription failed for %s: %s", filename, e)
        raise HTTPException(502, "Transcription service failed — try again shortly") from None

    if not transcript:
        return AudioIngestResponse(message="No speech detected in the recording")

    events = TextAdapter().ingest_text(
        transcript,
        metadata={"transcribed": True, "filename": filename},
        source=SourceType.CONVERSATION,
    )
    result = await run_in_threadpool(process_events, ws, events)
    return AudioIngestResponse(**result.model_dump(), transcript=transcript[:2000])
