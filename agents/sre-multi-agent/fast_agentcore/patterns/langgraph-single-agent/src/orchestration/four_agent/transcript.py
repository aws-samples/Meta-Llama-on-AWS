"""Transcript logging utilities for the four-agent orchestration demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from .schema import AgentMessage, TranscriptEntry
from ..observability import emit_event, wrap_payload


class TranscriptLogger:
    """Append-only JSONL writer that mirrors conversation traffic to disk."""

    def __init__(self, base_dir: Union[str, Path] = Path("logs/transcripts")) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, incident_id: str) -> Path:
        safe_id = incident_id.replace("/", "-").replace(" ", "_")
        return self.base_dir / f"{safe_id}.jsonl"

    def start_incident(self, incident_id: str) -> Path:
        """Ensure a transcript file exists for the provided incident."""

        path = self._path_for(incident_id)
        if not path.exists():
            path.touch()
            emit_event(
                "transcript_logger",
                "transcript_created",
                wrap_payload(incident_id=incident_id, path=str(path)),
            )
        return path

    def append(
        self,
        entry: Union[AgentMessage, TranscriptEntry],
        *,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Path:
        """Append a message to the transcript, returning the file path used."""

        if isinstance(entry, AgentMessage):
            # Start with caller-provided metadata (latency, stage, etc.)
            merged_meta: Dict[str, object] = dict(metadata or {})
            # If the message carries runtime metadata (e.g., LLM prompts/output),
            # merge it so transcripts preserve full context for audits.
            try:
                extra_meta = getattr(entry, "metadata", None)
                if isinstance(extra_meta, dict):
                    merged_meta.update(extra_meta)
            except Exception:
                # Never fail transcript logging because of metadata issues
                pass
            transcript_entry = TranscriptEntry(message=entry, metadata=merged_meta)
        else:
            transcript_entry = entry
            if metadata:
                transcript_entry.metadata.update(metadata)

        path = self.start_incident(transcript_entry.message.incident_id)

        with path.open("a", encoding="utf-8") as handle:
            handle.write(transcript_entry.model_dump_json(by_alias=True, indent=2) + "\n")

        msg = transcript_entry.message
        emit_event(
            "transcript_logger",
            "message_appended",
            wrap_payload(
                incident_id=msg.incident_id,
                sender=msg.sender.value if hasattr(msg.sender, "value") else str(msg.sender),
                type=msg.type.value if hasattr(msg.type, "value") else str(msg.type),
                severity=getattr(msg.severity, "value", None),
                path=str(path),
            ),
        )
        return path


def read_transcript(path: Union[str, Path]) -> List[TranscriptEntry]:
    """Utility helper for tests to load transcript contents."""

    resolved = Path(path)
    entries: List[TranscriptEntry] = []
    if not resolved.exists():
        return entries
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            entries.append(TranscriptEntry.model_validate(data))
    return entries


__all__ = ["TranscriptLogger", "read_transcript"]
