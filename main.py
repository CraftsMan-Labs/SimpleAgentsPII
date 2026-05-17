from __future__ import annotations
import asyncio
import logging
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from simple_agents_py import Client  # type: ignore[import-not-found]
from simple_agents_py.workflow_request import (
    WorkflowExecutionRequest,
    WorkflowMessage,
    WorkflowRole,
)  # type: ignore[import-not-found]
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

load_dotenv()
logger = logging.getLogger(__name__)

WORKFLOW_FILE = Path(__file__).resolve().parent / "workflow.yaml"
CHUNK_SIZE_LINES = 3
MAX_CONCURRENT_CHUNKS = 3
CHUNK_TIMEOUT_SECONDS = float(os.getenv("PII_CHUNK_TIMEOUT_SECONDS", "45"))
MAX_RETRY_ATTEMPTS = int(os.getenv("PII_RETRY_ATTEMPTS", "3"))

PIIType: TypeAlias = Literal[
    "PII_NAME",
    "PII_PHONE_NUMBER",
    "PII_EMAIL",
    "PII_ADDRESS",
    "PII_DOB",
    "PII_ID_NUMBER",
    "PII_BANK_ACCOUNT",
]
MappingDict: TypeAlias = dict[str, str]

BASE_PLACEHOLDER_PATTERN = re.compile(
    r"<(?P<pii_type>"
    r"PII_NAME|PII_PHONE_NUMBER|PII_EMAIL|PII_ADDRESS|"
    r"PII_DOB|PII_ID_NUMBER|PII_BANK_ACCOUNT"
    r")>"
)
ANY_PII_PLACEHOLDER_PATTERN = re.compile(r"<(?P<label>PII_[A-Z0-9_]+)>")
SUPPORTED_PII_TYPES = {
    "PII_NAME",
    "PII_PHONE_NUMBER",
    "PII_EMAIL",
    "PII_ADDRESS",
    "PII_DOB",
    "PII_ID_NUMBER",
    "PII_BANK_ACCOUNT",
}
PII_PLACEHOLDER_ALIASES = {
    "PII_PHONE": "PII_PHONE_NUMBER",
    "PII_MOBILE": "PII_PHONE_NUMBER",
    "PII_CONTACT_NUMBER": "PII_PHONE_NUMBER",
    "PII_FULL_NAME": "PII_NAME",
    "PII_PERSON_NAME": "PII_NAME",
    "PII_SSN": "PII_ID_NUMBER",
    "PII_TAX_ID": "PII_ID_NUMBER",
    "PII_IP_ADDRESS": "PII_ID_NUMBER",
    "PII_DATE": "PII_DOB",
    "PII_AMOUNT": "PII_BANK_ACCOUNT",
    "PII_BANK": "PII_BANK_ACCOUNT",
    "PII_ACCOUNT_NUMBER": "PII_BANK_ACCOUNT",
    "PII_ROUTING_NUMBER": "PII_BANK_ACCOUNT",
}
LINE_MARKER_PATTERN = re.compile(r"^__L(?P<idx>\d{4})__(?: (?P<text>.*))?$")
PII_SIGNAL_PATTERN = re.compile(
    r"@|"
    r"\d{2,}|"
    r"\b(name|phone|email|address|dob|id|account|card|passport|license|routing|"
    r"beneficiary|holder|customer|tax|ssn)\b",
    flags=re.IGNORECASE,
)


class RedactRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Raw input text to redact.",
    )


class ChunkOutput(BaseModel):
    redacted_text: str


class RedactResponse(BaseModel):
    redacted_text: str
    mapping_keys: MappingDict
    total_chunks: int
    succeeded_chunks: int
    failed_chunks: list[dict[str, Any]]


class TransientWorkflowError(RuntimeError):
    """Retryable workflow or provider error."""


class InvalidChunkOutputError(RuntimeError):
    """Chunk output is malformed and cannot be used safely."""


@dataclass(frozen=True)
class PlaceholderOccurrence:
    pii_type: PIIType
    original_value: str


app = FastAPI(title="PII Redaction Service")


def _build_client() -> Client:
    provider = os.getenv("WORKFLOW_PROVIDER")
    api_base = os.getenv("WORKFLOW_API_BASE")
    api_key = os.getenv("WORKFLOW_API_KEY")
    if not provider or not api_base or not api_key:
        raise RuntimeError(
            "Missing required environment variables: "
            "WORKFLOW_PROVIDER, WORKFLOW_API_BASE, WORKFLOW_API_KEY"
        )
    return Client(provider, api_base=api_base, api_key=api_key)


def _split_text_into_chunks(
    text: str,
    chunk_size_lines: int = CHUNK_SIZE_LINES,
) -> list[str]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return [text]
    return [
        "".join(lines[idx: idx + chunk_size_lines])
        for idx in range(0, len(lines), chunk_size_lines)
    ]


def _line_count(value: str) -> int:
    return len(value.splitlines())


def _split_line_body_and_ending(line: str) -> tuple[str, str]:
    body = line.rstrip("\r\n")
    ending = line[len(body) :]
    return body, ending


def _split_lines_keepends(value: str) -> list[str]:
    lines = value.splitlines(keepends=True)
    if not lines:
        return [value]
    return lines


def _chunk_needs_llm(chunk_text: str) -> bool:
    if not chunk_text.strip():
        return False
    return bool(PII_SIGNAL_PATTERN.search(chunk_text))


def _encode_chunk_with_markers(chunk_text: str) -> str:
    lines = _split_lines_keepends(chunk_text)
    encoded_lines: list[str] = []
    for idx, line in enumerate(lines, start=1):
        body, _ = _split_line_body_and_ending(line)
        encoded_lines.append(
            f"__L{idx:04d}__{(' ' + body) if body else ''}"
        )
    return "\n".join(encoded_lines)


def _extract_marked_lines(model_output: str, expected_count: int) -> list[str] | None:
    matched: dict[int, str] = {}
    for line in model_output.splitlines():
        match = LINE_MARKER_PATTERN.match(line)
        if match is None:
            continue
        idx = int(match.group("idx"))
        if 1 <= idx <= expected_count and idx not in matched:
            matched[idx] = match.group("text") or ""

    if len(matched) != expected_count:
        return None
    return [matched[idx] for idx in range(1, expected_count + 1)]


def _restore_chunk_line_structure(original_chunk: str, model_output: str) -> str:
    original_lines = _split_lines_keepends(original_chunk)
    expected_count = len(original_lines)

    marker_lines = _extract_marked_lines(model_output, expected_count)
    if marker_lines is not None:
        rebuilt_parts: list[str] = []
        for source_line, marked_text in zip(original_lines, marker_lines, strict=True):
            _, ending = _split_line_body_and_ending(source_line)
            rebuilt_parts.append(f"{marked_text}{ending}")
        return "".join(rebuilt_parts)

    output_lines = model_output.splitlines()
    if len(output_lines) == expected_count:
        rebuilt_parts = []
        for source_line, out_line in zip(original_lines, output_lines, strict=True):
            _, ending = _split_line_body_and_ending(source_line)
            rebuilt_parts.append(f"{out_line}{ending}")
        return "".join(rebuilt_parts)

    original_non_empty_positions = [
        idx
        for idx, line in enumerate(original_lines)
        if _split_line_body_and_ending(line)[0].strip()
    ]
    output_non_empty_lines = [line for line in output_lines if line.strip()]
    if len(output_non_empty_lines) == len(original_non_empty_positions):
        recovered = ["" for _ in range(expected_count)]
        for pos, out_line in zip(original_non_empty_positions, output_non_empty_lines, strict=True):
            recovered[pos] = out_line

        rebuilt_parts = []
        for source_line, recovered_line in zip(original_lines, recovered, strict=True):
            _, ending = _split_line_body_and_ending(source_line)
            rebuilt_parts.append(f"{recovered_line}{ending}")
        return "".join(rebuilt_parts)

    raise InvalidChunkOutputError(
        "Could not restore output to original chunk line structure."
    )


def _log_chunk_failure(
    *,
    chunk_index: int | None,
    reason: str,
    input_chunk: str,
    output_snapshot: Any | None = None,
) -> None:
    logger.error(
        "PII chunk failure | chunk_index=%s | reason=%s\n"
        "INPUT_CHUNK_START\n%s\nINPUT_CHUNK_END\n"
        "OUTPUT_SNAPSHOT_START\n%s\nOUTPUT_SNAPSHOT_END",
        chunk_index if chunk_index is not None else "unknown",
        reason,
        input_chunk,
        output_snapshot if output_snapshot is not None else "<none>",
    )


def _iter_dict_values(data: Any) -> Iterable[Any]:
    if isinstance(data, dict):
        for value in data.values():
            yield value
    elif isinstance(data, list):
        for item in data:
            yield item


def _find_chunk_output_payload(data: Any) -> Any | None:
    if isinstance(data, dict):
        # Prefer well-known terminal/output fields first.
        for key in ("terminal_output", "output", "redacted_text"):
            value = data.get(key)
            if isinstance(value, str):
                return value

        if "redacted_lines" in data:
            return data

    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        return data

    for value in _iter_dict_values(data):
        if isinstance(value, str):
            continue
        match = _find_chunk_output_payload(value)
        if match is not None:
            return match
    return None


def _validate_placeholders(redacted_text: str) -> None:
    raw_placeholders = re.findall(r"<([A-Z0-9_]+)>", redacted_text)
    for raw in raw_placeholders:
        if raw.startswith("PII_") and raw not in SUPPORTED_PII_TYPES:
            raise InvalidChunkOutputError(
                f"Unsupported placeholder '<{raw}>' in chunk output."
            )


def _normalize_placeholder_labels(redacted_text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        label = match.group("label")
        if label in SUPPORTED_PII_TYPES:
            return match.group(0)
        canonical = PII_PLACEHOLDER_ALIASES.get(label, "PII_ID_NUMBER")
        return f"<{canonical}>"

    return ANY_PII_PLACEHOLDER_PATTERN.sub(replace, redacted_text)


def _extract_occurrences_from_chunk(
    original_text: str,
    redacted_text: str,
) -> list[PlaceholderOccurrence]:
    split_parts = BASE_PLACEHOLDER_PATTERN.split(redacted_text)
    if len(split_parts) == 1:
        if original_text != redacted_text:
            raise InvalidChunkOutputError(
                "Chunk text changed without valid placeholder replacement."
            )
        return []

    first_literal = split_parts[0]
    cursor = 0
    if first_literal:
        first_pos = original_text.find(first_literal, cursor)
        if first_pos < 0:
            raise InvalidChunkOutputError(
                "Could not align first literal segment with source text."
            )
        cursor = first_pos + len(first_literal)

    occurrences: list[PlaceholderOccurrence] = []
    placeholder_count = (len(split_parts) - 1) // 2

    for idx in range(placeholder_count):
        pii_type_str = split_parts[(2 * idx) + 1]
        next_literal = split_parts[(2 * idx) + 2]
        pii_type = pii_type_str  # narrowed by regex split

        if next_literal:
            next_pos = original_text.find(next_literal, cursor)
            if next_pos < 0:
                raise InvalidChunkOutputError(
                    "Could not align redacted placeholder boundaries with source text."
                )
            original_value = original_text[cursor:next_pos]
            cursor = next_pos + len(next_literal)
        else:
            original_value = original_text[cursor:]
            cursor = len(original_text)

        if not original_value:
            raise InvalidChunkOutputError(
                f"Placeholder <{pii_type}> has empty replacement in aligned source text."
            )

        occurrences.append(
            PlaceholderOccurrence(
                pii_type=pii_type,  # type: ignore[arg-type]
                original_value=original_value,
            )
        )

    return occurrences


def _normalize_chunk_output(payload: Any, chunk_text: str) -> ChunkOutput:
    if isinstance(payload, str):
        return ChunkOutput(redacted_text=payload)

    if isinstance(payload, dict):
        try:
            parsed = ChunkOutput.model_validate(payload)
            return parsed
        except Exception:
            redacted_lines = payload.get("redacted_lines")
    elif isinstance(payload, list) and all(isinstance(item, str) for item in payload):
        redacted_lines = payload
    else:
        redacted_lines = None

    if not isinstance(redacted_lines, list) or not all(isinstance(item, str) for item in redacted_lines):
        raise InvalidChunkOutputError("Chunk output must be a redacted_text object or string array.")

    chunk_lines = _split_lines_keepends(chunk_text)

    if len(redacted_lines) != len(chunk_lines):
        raise InvalidChunkOutputError(
            "Line count mismatch for chunk: "
            f"input={len(chunk_lines)}, output={len(redacted_lines)}"
        )

    rebuilt_parts: list[str] = []
    for source_line, redacted_line in zip(chunk_lines, redacted_lines, strict=True):
        line_ending = source_line[len(source_line.rstrip("\r\n")) :]
        cleaned = redacted_line.rstrip("\r\n")
        rebuilt_parts.append(f"{cleaned}{line_ending}")

    return ChunkOutput(redacted_text="".join(rebuilt_parts))


@retry(
    retry=retry_if_exception_type((TransientWorkflowError, InvalidChunkOutputError)),
    wait=wait_exponential_jitter(initial=1, max=8),
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _run_redaction_chunk(chunk_text: str, chunk_index: int | None = None) -> ChunkOutput:
    if not _chunk_needs_llm(chunk_text):
        return ChunkOutput(redacted_text=chunk_text)

    llm_input = _encode_chunk_with_markers(chunk_text)
    req = WorkflowExecutionRequest(
        workflow_path=str(WORKFLOW_FILE),
        messages=[WorkflowMessage(role=WorkflowRole.USER, content=llm_input)],
    )
    try:
        raw_result = _build_client().run_workflow(req)
    except Exception as exc:
        _log_chunk_failure(
            chunk_index=chunk_index,
            reason=f"workflow call failed: {exc}",
            input_chunk=llm_input,
        )
        raise TransientWorkflowError(f"Workflow execution failed: {exc}") from exc

    wire_result = raw_result.to_dict() if hasattr(raw_result, "to_dict") else raw_result
    payload = _find_chunk_output_payload(wire_result)
    if payload is None:
        _log_chunk_failure(
            chunk_index=chunk_index,
            reason="missing output payload in workflow result",
            input_chunk=llm_input,
            output_snapshot=wire_result,
        )
        raise InvalidChunkOutputError(
            "Could not locate redaction payload in workflow result."
        )

    try:
        parsed = _normalize_chunk_output(payload, chunk_text)
    except Exception as exc:
        _log_chunk_failure(
            chunk_index=chunk_index,
            reason=f"output normalization failed: {exc}",
            input_chunk=llm_input,
            output_snapshot=payload,
        )
        raise

    try:
        restored_text = _restore_chunk_line_structure(chunk_text, parsed.redacted_text)
    except Exception as exc:
        _log_chunk_failure(
            chunk_index=chunk_index,
            reason=f"line-structure restoration failed: {exc}",
            input_chunk=llm_input,
            output_snapshot=parsed.redacted_text,
        )
        raise

    input_line_count = _line_count(chunk_text)
    output_line_count = _line_count(restored_text)
    if input_line_count != output_line_count:
        _log_chunk_failure(
            chunk_index=chunk_index,
            reason=(
                "line-count mismatch after normalization: "
                f"input={input_line_count}, output={output_line_count}"
            ),
            input_chunk=llm_input,
            output_snapshot=restored_text,
        )
        raise InvalidChunkOutputError(
            "Line count mismatch for chunk: "
            f"input={input_line_count}, output={output_line_count}"
        )

    normalized_text = _normalize_placeholder_labels(restored_text)
    try:
        _validate_placeholders(normalized_text)
    except Exception as exc:
        _log_chunk_failure(
            chunk_index=chunk_index,
            reason=f"placeholder validation failed: {exc}",
            input_chunk=llm_input,
            output_snapshot=normalized_text,
        )
        raise

    return ChunkOutput(redacted_text=normalized_text)


async def _process_chunk(
    index: int,
    chunk_text: str,
    semaphore: asyncio.Semaphore,
) -> tuple[int, ChunkOutput]:
    async with semaphore:
        try:
            chunk_output = await asyncio.wait_for(
                asyncio.to_thread(_run_redaction_chunk, chunk_text, index),
                timeout=CHUNK_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            raise TransientWorkflowError(
                f"Timed out while processing chunk {index} "
                f"after {CHUNK_TIMEOUT_SECONDS} seconds"
            ) from exc
    return index, chunk_output


def _apply_indexed_placeholders(
    base_redacted_text: str,
    occurrences: list[PlaceholderOccurrence],
) -> tuple[str, MappingDict]:
    next_placeholder_number: dict[str, int] = defaultdict(int)
    mapping: MappingDict = {}
    cursor = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal cursor
        if cursor >= len(occurrences):
            raise InvalidChunkOutputError(
                "Redacted text contains more placeholders than extracted source values."
            )

        pii_type = match.group("pii_type")
        occurrence = occurrences[cursor]
        cursor += 1
        if occurrence.pii_type != pii_type:
            raise InvalidChunkOutputError(
                f"Placeholder type mismatch: expected {occurrence.pii_type}, got {pii_type}."
            )

        next_placeholder_number[pii_type] += 1
        key = f"{pii_type}_{next_placeholder_number[pii_type]}"
        mapping[key] = occurrence.original_value
        return f"<{key}>"

    indexed_text = BASE_PLACEHOLDER_PATTERN.sub(replace, base_redacted_text)
    if cursor != len(occurrences):
        raise InvalidChunkOutputError(
            "Extracted source values contain more entries than redacted placeholders."
        )
    return indexed_text, mapping


@app.post("/redact", response_model=RedactResponse)
async def redact_pii(payload: RedactRequest) -> RedactResponse:
    return await _redact_text(payload.text)


async def _redact_text(text: str) -> RedactResponse:
    chunks = _split_text_into_chunks(text, CHUNK_SIZE_LINES)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)
    tasks = [
        asyncio.create_task(
            _process_chunk(
                index=idx,
                chunk_text=chunk,
                semaphore=semaphore,
            )
        )
        for idx, chunk in enumerate(chunks)
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    failed_chunks: list[dict[str, Any]] = []
    successful_outputs: dict[int, ChunkOutput] = {}
    for idx, result in enumerate(raw_results):
        if isinstance(result, Exception):
            failed_chunks.append(
                {
                    "chunk_index": idx,
                    "stage": "llm_execution",
                    "error": str(result),
                }
            )
            continue
        chunk_index, chunk_output = result
        successful_outputs[chunk_index] = chunk_output

    rendered_chunks = list(chunks)
    occurrences: list[PlaceholderOccurrence] = []
    for chunk_index in range(len(chunks)):
        if chunk_index not in successful_outputs:
            continue
        chunk_output = successful_outputs[chunk_index]
        chunk_source = chunks[chunk_index]
        try:
            chunk_occurrences = _extract_occurrences_from_chunk(
                original_text=chunk_source,
                redacted_text=chunk_output.redacted_text,
            )
        except Exception as exc:
            failed_chunks.append(
                {
                    "chunk_index": chunk_index,
                    "stage": "postprocess_alignment",
                    "error": str(exc),
                }
            )
            rendered_chunks[chunk_index] = chunk_source
            continue

        rendered_chunks[chunk_index] = chunk_output.redacted_text
        occurrences.extend(chunk_occurrences)

    base_redacted_text = "".join(rendered_chunks)

    try:
        final_text, mapping = _apply_indexed_placeholders(
            base_redacted_text,
            occurrences,
        )
    except Exception as exc:
        failed_chunks.append(
            {
                "chunk_index": -1,
                "stage": "global_placeholder_indexing",
                "error": str(exc),
            }
        )
        final_text = base_redacted_text
        mapping = {}

    succeeded_chunks = len(chunks) - len(
        {item["chunk_index"] for item in failed_chunks if item["chunk_index"] >= 0}
    )
    return RedactResponse(
        redacted_text=final_text,
        mapping_keys=mapping,
        total_chunks=len(chunks),
        succeeded_chunks=succeeded_chunks,
        failed_chunks=failed_chunks,
    )


@app.post("/redact-file", response_model=RedactResponse)
async def redact_pii_file(file: UploadFile = File(...)) -> RedactResponse:
    filename = (file.filename or "").lower()
    if not filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are supported.")

    raw_bytes = await file.read()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded markdown file must be UTF-8 encoded.",
        ) from exc

    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded markdown file is empty.")

    return await _redact_text(text)
