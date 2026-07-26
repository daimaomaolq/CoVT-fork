from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .geometry import clamp_box
from .schema import Candidate, Observation


FORBIDDEN_QUERY_FIELDS = {"question_e", "question_e_cn"}


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def inference_input_from_row(
    row: dict[str, Any],
    query_field: str = "question",
) -> dict[str, str]:
    """Create the only dictionary that may cross into the inference system.

    The explicit whitelist prevents GT, IoU, dataset-side oracle rewrites, and
    other evaluation metadata from becoming accidentally visible to the parent.
    """
    normalized_field = str(query_field).strip().lower()
    if normalized_field in FORBIDDEN_QUERY_FIELDS:
        raise ValueError(
            f"{query_field!r} is an oracle-style alternate query and is forbidden "
            "for the main agentic inference track"
        )
    if normalized_field not in {"question", "question_cn", "query"}:
        raise ValueError(
            "query_field must be one of question, question_cn, or a verified query"
        )
    query = str(row.get(query_field) or "").strip()
    if not query:
        raise ValueError(f"Missing non-empty query field {query_field!r}")
    if normalized_field == "query":
        provenance = " ".join(
            str(row.get(key) or "")
            for key in ("query_rule", "query_version", "task_tag")
        ).lower()
        if "question_e" in provenance:
            raise ValueError(
                "The generic query field is derived from forbidden question_e"
            )
        oracle_text = str(row.get("question_e") or "").strip()
        ordinary_text = str(row.get("question") or "").strip()
        if oracle_text and query == oracle_text and query != ordinary_text:
            raise ValueError(
                "The generic query field exactly matches forbidden question_e"
            )
    sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
    image = str(row.get("image") or row.get("image_path") or "").strip()
    if not sample_id:
        raise ValueError("Missing sample_id/id")
    if not image:
        raise ValueError(f"Missing image path for sample {sample_id}")
    return {
        "sample_id": sample_id,
        "image": image,
        "query": query,
        "class": str(row.get("class") or row.get("category") or "unknown").strip()
        or "unknown",
    }


def resolve_image_path(value: str, index_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = index_path.parent / path
    return path.resolve()


def _nested_get(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _first_not_none(values: Iterable[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _has_prediction_provenance(row: dict[str, Any]) -> bool:
    if isinstance(row.get("inference"), dict):
        return True
    markers = {
        "pred_bbox",
        "final_bbox",
        "prediction",
        "raw_output",
        "parse_ok",
        "bbox_token_confidence",
        "generation_confidence",
    }
    return any(marker in row for marker in markers)


def candidate_from_prediction(row: dict[str, Any], query: str = "") -> Candidate:
    candidate_record = _nested_get(row, ("inference", "initial_candidate"))
    if not isinstance(candidate_record, dict):
        candidate_record = {}
    bbox = _first_not_none(
        (
            candidate_record.get("bbox"),
            row.get("pred_bbox"),
            row.get("bbox"),
            row.get("final_bbox"),
            _nested_get(row, ("inference", "final_bbox")),
        )
    )
    confidence = _first_not_none(
        (
            row.get("bbox_token_confidence"),
            row.get("generation_confidence"),
            candidate_record.get("bbox_token_confidence"),
        )
    )
    explicit_availability = _first_not_none(
        (
            row.get("confidence_available"),
            candidate_record.get("confidence_available"),
        )
    )
    confidence_available = (
        bool(explicit_availability)
        if explicit_availability is not None
        else confidence is not None
    )
    if confidence is None:
        confidence = 0.5
    parsed_bbox = None
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        values = [float(value) for value in bbox[:4]]
        if max(abs(value) for value in values) > 1.5:
            values = [value / 1000.0 for value in values]
        parsed_bbox = clamp_box(values)
    raw_output = str(
        _first_not_none((row.get("raw_output"), candidate_record.get("raw_output"), ""))
    )
    return Candidate(
        candidate_id="c00",
        bbox=parsed_bbox,
        source_agent="BaseGrounder",
        query_used=query,
        observation=Observation(
            observation_id="full",
            view_type="full_image",
            preserves_context=True,
        ),
        bbox_token_confidence=max(0.0, min(1.0, float(confidence))),
        confidence_available=confidence_available,
        bbox_token_count=int(candidate_record.get("bbox_token_count") or 0),
        raw_output=raw_output,
        latency_ms=float(
            _first_not_none(
                (
                    candidate_record.get("latency_ms"),
                    _nested_get(row, ("cost", "initial_latency_ms")),
                    row.get("latency_ms"),
                    0.0,
                )
            )
            or 0.0
        ),
    )


def load_cached_predictions(
    path: Path | None,
    require_confidence: bool = False,
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
        if not sample_id:
            raise ValueError(f"Cached prediction row in {path} has no sample_id/id")
        if not _has_prediction_provenance(row):
            raise ValueError(
                f"Cached row {sample_id} in {path} has no prediction provenance; "
                "refusing a possible dataset-index/GT row"
            )
        if require_confidence:
            candidate = candidate_from_prediction(row)
            if not candidate.confidence_available or candidate.bbox_token_count <= 0:
                raise ValueError(
                    f"Cached prediction {sample_id} in {path} has no measured bbox-token confidence. "
                    "Generate the formal one-pass cache before running the agent matrix."
                )
        if sample_id in result:
            raise ValueError(f"Duplicate cached prediction for sample_id={sample_id}")
        result[sample_id] = row
    return result
