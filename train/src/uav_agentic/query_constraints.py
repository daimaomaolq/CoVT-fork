from __future__ import annotations

import re

from .schema import QueryConstraintGraph, SpatialFrame


TARGET_LEXICON = (
    "three-wheeled vehicle",
    "traffic light",
    "tennis court",
    "parking lot",
    "construction vehicle",
    "motorcycle",
    "motorbike",
    "bicycle",
    "tricycle",
    "airplane",
    "aircraft",
    "plane",
    "helicopter",
    "vehicle",
    "truck",
    "lorry",
    "bus",
    "van",
    "car",
    "person",
    "pedestrian",
    "man",
    "woman",
    "child",
    "worker",
    "cyclist",
    "rider",
    "boat",
    "ship",
    "building",
    "house",
    "tent",
    "container",
    "excavator",
    "tractor",
    "crosswalk",
    "zebra crossing",
    "supermarket",
    "farmland",
    "land",
    "formation",
    "place",
    "area",
    "region",
    "field",
    "stage",
    "road",
    "street",
    "bridge",
    "viaduct",
    "fence",
    "playground",
    "goal",
    "player",
)

ATTRIBUTE_TERMS = (
    "white",
    "black",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "gray",
    "grey",
    "large",
    "small",
    "tiny",
    "parked",
    "moving",
    "standing",
    "illegal",
    "three-wheeled",
    "two-wheeled",
    "striped",
    "damaged",
)

RELATION_PATTERNS = (
    ("front", r"\bin front of\b|\bahead of\b"),
    ("behind", r"\bbehind\b"),
    ("left", r"\bleft of\b"),
    ("right", r"\bright of\b"),
    ("above", r"\babove\b|\bover\b"),
    ("below", r"\bbelow\b|\bunder\b"),
    ("near", r"\bnear\b|\bnext to\b|\bbeside\b|\bclose to\b|\badjacent to\b"),
    ("inside", r"\binside\b|\bwithin\b"),
    ("overlap", r"\bparked on\b|\bstanding on\b|\boccupying\b|\boverlapping\b"),
)

CONTEXT_SPLIT_PATTERNS = (
    r"\bin front of\b",
    r"\bahead of\b",
    r"\bleft of\b",
    r"\bright of\b",
    r"\bnext to\b",
    r"\bclose to\b",
    r"\bparked on\b",
    r"\bstanding on\b",
    r"\bbehind\b",
    r"\bbeside\b",
    r"\badjacent to\b",
    r"\binside\b",
    r"\bwithin\b",
    r"\babove\b",
    r"\bbelow\b",
    r"\bunder\b",
    r"\bnear\b",
)

GLOBAL_POSITION_PATTERNS = (
    ("upper_left", r"\bupper[- ]left\b|\btop[- ]left\b|\bleft upper\b"),
    ("upper_right", r"\bupper[- ]right\b|\btop[- ]right\b|\bright upper\b"),
    ("lower_left", r"\blower[- ]left\b|\bbottom[- ]left\b|\bleft lower\b"),
    ("lower_right", r"\blower[- ]right\b|\bbottom[- ]right\b|\bright lower\b"),
    ("left", r"\bon the left\b|\bleft side\b"),
    ("right", r"\bon the right\b|\bright side\b"),
    ("top", r"\bat the top\b|\btop side\b|\bupper part\b"),
    ("bottom", r"\bat the bottom\b|\bbottom side\b|\blower part\b"),
    ("center", r"\bin the middle\b|\bin the center\b|\bcentral\b"),
    ("edge", r"\bat the edge\b|\bnear the boundary\b"),
)

ORDINAL_PATTERNS = (
    ("leftmost", r"\bleftmost\b|\bfurthest left\b"),
    ("rightmost", r"\brightmost\b|\bfurthest right\b"),
    ("topmost", r"\btopmost\b|\bhighest\b"),
    ("bottommost", r"\bbottommost\b|\blowest\b"),
    ("first", r"\bfirst\b"),
    ("second", r"\bsecond\b"),
    ("third", r"\bthird\b"),
)

TEMPORAL_PATTERNS = (
    r"\babout to\b",
    r"\bgoing to\b",
    r"\bcurrently\b",
    r"\bmoving\b",
    r"\bleaving\b",
    r"\bmerging\b",
    r"\bentering\b",
    r"\bexiting\b",
)

ORIENTATION_PATTERNS = (
    r"\bfacing\b",
    r"\bheading\b",
    r"\bfront of\b",
    r"\bbehind\b",
    r"\btoward\b",
    r"\btowards\b",
)

ARTICLE_PATTERN = re.compile(r"^(the|a|an)\s+", flags=re.IGNORECASE)


def _first_match(patterns: tuple[tuple[str, str], ...], text: str) -> str | None:
    for name, pattern in patterns:
        if re.search(pattern, text):
            return name
    return None


def _extract_target(lower: str) -> str:
    matches: list[tuple[int, int, str]] = []
    for target in TARGET_LEXICON:
        match = re.search(r"\b" + re.escape(target) + r"\b", lower)
        if match:
            matches.append((match.start(), -len(target), target))
    if matches:
        return sorted(matches)[0][2]
    words = re.findall(r"[a-z0-9-]+", lower)
    stopwords = {
        "the",
        "a",
        "an",
        "that",
        "which",
        "is",
        "are",
        "in",
        "on",
        "of",
        "with",
        "to",
        "from",
        "near",
        "left",
        "right",
        "upper",
        "lower",
        "middle",
        "front",
        "behind",
        "above",
        "below",
        "under",
    }
    content = [word for word in words if word not in stopwords]
    return " ".join(content[-3:]) if content else lower


def _first_context_split(lower: str) -> re.Match[str] | None:
    best: re.Match[str] | None = None
    for pattern in CONTEXT_SPLIT_PATTERNS:
        match = re.search(pattern, lower)
        if match and (best is None or match.start() < best.start()):
            best = match
    return best


def _extract_context(original: str, lower: str) -> str:
    positional_only = bool(re.search(r"\bon the (left|right)\b", lower))
    if positional_only:
        return ""
    split = _first_context_split(lower)
    if split is None:
        return ""
    context = original[split.end() :].strip(" ,.;:")
    context = ARTICLE_PATTERN.sub("", context)
    return context


def _remove_global_phrases(text: str) -> str:
    result = text
    for _, pattern in GLOBAL_POSITION_PATTERNS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    for _, pattern in ORDINAL_PATTERNS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    result = re.sub(
        r"\b(?:in|on|at|near)\s+(?:the\s*)?$",
        "",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"\s+", " ", result).strip(" ,.;:")
    return result


def _extract_target_clause(original: str, lower: str) -> str:
    split = _first_context_split(lower)
    clause = original[: split.start()] if split is not None else original
    clause = ARTICLE_PATTERN.sub("", clause.strip(" ,.;:"))
    clause = _remove_global_phrases(clause)
    clause = re.sub(
        r"\b(?:who|that|which)\s+(?:is|are)\s+(?:temporarily|currently)\s*$",
        "",
        clause,
        flags=re.IGNORECASE,
    ).strip(" ,.;:")
    return clause or original


def parse_query_constraints(query: str) -> QueryConstraintGraph:
    original = " ".join(str(query).strip().split())
    lower = original.lower()
    target_clause = _extract_target_clause(original, lower)
    target_clause_lower = target_clause.lower()
    target = _extract_target(target_clause_lower)
    attributes = [
        attribute
        for attribute in ATTRIBUTE_TERMS
        if re.search(r"\b" + re.escape(attribute) + r"\b", target_clause_lower)
    ]
    relations = [
        name for name, pattern in RELATION_PATTERNS if re.search(pattern, lower)
    ]
    context = _extract_context(original, lower)
    global_position = _first_match(GLOBAL_POSITION_PATTERNS, lower)
    ordinal = _first_match(ORDINAL_PATTERNS, lower)

    frames: list[SpatialFrame] = []
    if global_position:
        frames.append(SpatialFrame.GLOBAL_ABSOLUTE)
    if ordinal:
        frames.append(SpatialFrame.GLOBAL_ORDER)
    if context or relations:
        frames.append(SpatialFrame.OBJECT_RELATIVE)
    if any(re.search(pattern, lower) for pattern in ORIENTATION_PATTERNS):
        frames.append(SpatialFrame.ORIENTATION_DEPENDENT)
    if any(re.search(pattern, lower) for pattern in TEMPORAL_PATTERNS):
        frames.append(SpatialFrame.TEMPORAL_EVENT)
    if not frames:
        frames.append(SpatialFrame.LOCAL_ATTRIBUTE)

    local_target_query = target_clause
    if SpatialFrame.GLOBAL_ABSOLUTE in frames or SpatialFrame.GLOBAL_ORDER in frames:
        zoom_query = local_target_query
    elif SpatialFrame.OBJECT_RELATIVE in frames:
        zoom_query = _remove_global_phrases(original)
    else:
        zoom_query = original
    return QueryConstraintGraph(
        original=original,
        target=target,
        attributes=attributes,
        context=context,
        relations=relations,
        spatial_frames=list(dict.fromkeys(frames)),
        global_position=global_position,
        ordinal_constraint=ordinal,
        local_target_query=local_target_query,
        zoom_query=zoom_query or local_target_query,
    )
