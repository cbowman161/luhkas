"""LLM output normalization utilities.

The normalizer is intentionally forgiving. It accepts common local-model drift
(prose wrappers, markdown decoration, fenced code blocks, FILE headers, path
labels, duplicated RAW OUTPUT sections, and terminal control-character noise)
and converts that mess into strict internal file objects.

The validator remains strict and is the only gate before files are written.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Tuple

from .schemas import BuildFiles, GeneratedFile

# ANSI CSI such as ESC[3D, ESC[K, ESC[1;32m. Kept for regex fallback cleanup.
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# Remove non-printing control chars except tab/newline. ESC is handled separately.
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1a\x1c-\x1f\x7f]")

VALID_PATH_RE = re.compile(r"(?:src|tests|artifacts)/[A-Za-z0-9_./-]+")

# Flexible FILE envelope. Handles:
#   FILE: src/api.py
#   **FILE: src/api.py**
#   `FILE: src/api.py`
#   ### FILE: src/api.py
# Optional metadata lines are allowed before BEGIN.
FILE_HEADER_RE = re.compile(
    r"(?im)^\s*(?:[#>*_`\-\s]*)FILE\s*:\s*(?P<path>(?:src|tests|artifacts)/[^\n*`]+?)(?:\s*[*_`#-]*)?\s*$"
)
BEGIN_RE = re.compile(r"(?im)^\s*(?:[*_`\s]*)---BEGIN\s+CONTENT---(?:[*_`\s]*)$", re.MULTILINE)
END_RE = re.compile(r"(?im)^\s*(?:[*_`\s]*)---END\s+CONTENT---(?:[*_`\s]*)$", re.MULTILINE)

# Path immediately followed by fenced block:
#   src/api.py
#   ```python
#   ...
#   ```
PATH_FENCE_RE = re.compile(
    r"(?im)^\s*(?:[*_`#>\-\s]*)(?P<path>(?:src|tests|artifacts)/[^\n`:'\"]+)(?:\s*[*_`#>\-]*)?:?\s*$\s*\n"
    r"\s*```(?P<lang>[A-Za-z0-9_+.-]*)?\s*\n"
    r"(?P<content>[\s\S]*?)\n\s*```",
    re.MULTILINE,
)

# Path: ... / Content: ... blocks.
PATH_CONTENT_RE = re.compile(
    r"(?im)^\s*(?:PATH|Path|path)\s*:\s*(?P<path>(?:src|tests|artifacts)/[^\n]+?)\s*$\n"
    r"(?:\s*CONTENT_TYPE\s*:\s*[^\n]*\n)?"
    r"\s*(?:CONTENT|Content|content)\s*:\s*$\n"
    r"(?P<content>[\s\S]*?)"
    r"(?=\n\s*(?:PATH|Path|path)\s*:\s*(?:src|tests|artifacts)/|\n\s*(?:[#>*_`\-\s]*)FILE\s*:|\Z)",
    re.MULTILINE,
)

# A last-resort extractor for sections starting at a path/header and ending at
# the next file header/path or prose footer.
SECTION_START_RE = re.compile(
    r"(?im)^\s*(?:[*_`#>\-\s]*(?:FILE\s*:\s*)?)(?P<path>(?:src|tests|artifacts)/[A-Za-z0-9_./-]+)(?:\s*[*_`#>\-]*)?:?\s*$"
)


def clean_model_text(text: str) -> str:
    """Remove terminal noise while preserving useful code text.

    This includes a tiny terminal-control emulator for the most common local
    model artifact: cursor-left + erase-line sequences like ESC[3D ESC[K.
    After emulation, remaining ANSI/control bytes are stripped.
    """
    text = _emulate_common_ansi(text or "")
    text = ANSI_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse excessive trailing whitespace per line without changing indentation.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text


def normalize_generated_files(raw: str) -> BuildFiles:
    """Normalize messy LLM output into BuildFiles.

    Accepted forms include:
    - FILE: path + ---BEGIN CONTENT--- blocks, with markdown decoration
    - path line followed by fenced Python/code block
    - Path: path / Content: blocks
    - JSON object/list containing files with path/content keys
    - section extraction from repeated/prose-wrapped output
    """
    text = clean_model_text(raw)
    candidates: List[Tuple[str, GeneratedFile]] = []

    for source, extractor in [
        ("file_envelope", _extract_file_envelopes),
        ("path_fenced_block", _extract_path_fenced_blocks),
        ("path_content_block", _extract_path_content_blocks),
        ("json_files", _extract_json_files),
        ("section_scan", _extract_sections_by_headers),
    ]:
        for file in extractor(text):
            candidates.append((source, file))

    files, sources = _dedupe_files(candidates)
    if not files:
        raise ValueError(f"No generated files could be normalized from LLM output\n\nRAW OUTPUT:\n{raw}")

    return BuildFiles(files=files, notes="normalized_from=" + ",".join(sources))


def extract_json_contract(raw: str, label: str = "model") -> Dict[str, Any]:
    cleaned = clean_model_text(raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        extracted = _extract_first_json_object(cleaned)
        try:
            return json.loads(extracted)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from {label}: {exc}\n\nRAW OUTPUT:\n{raw}") from exc


def _extract_file_envelopes(text: str) -> List[GeneratedFile]:
    files: List[GeneratedFile] = []
    headers = list(FILE_HEADER_RE.finditer(text))
    for header in headers:
        path = _clean_path(header.group("path"))
        if not path:
            continue
        begin = BEGIN_RE.search(text, header.end())
        if not begin:
            continue
        # If another FILE header appears before BEGIN, this header is not an envelope.
        next_header = FILE_HEADER_RE.search(text, header.end())
        if next_header and next_header.start() < begin.start():
            continue
        end = END_RE.search(text, begin.end())
        if not end:
            continue
        content = _clean_file_content(path, text[begin.end():end.start()])
        if content:
            files.append(GeneratedFile(path=path, content=content))
    return files


def _extract_path_fenced_blocks(text: str) -> List[GeneratedFile]:
    files: List[GeneratedFile] = []
    for match in PATH_FENCE_RE.finditer(text):
        path = _clean_path(match.group("path"))
        content = _clean_file_content(path, match.group("content"))
        if path and content:
            files.append(GeneratedFile(path=path, content=content))
    return files


def _extract_path_content_blocks(text: str) -> List[GeneratedFile]:
    files: List[GeneratedFile] = []
    for match in PATH_CONTENT_RE.finditer(text):
        path = _clean_path(match.group("path"))
        content = _clean_file_content(path, match.group("content"))
        if path and content:
            files.append(GeneratedFile(path=path, content=content))
    return files


def _extract_sections_by_headers(text: str) -> List[GeneratedFile]:
    """Last-resort extraction: locate file section headers and trim useful body.

    This handles outputs like:
        **FILE: src/api.py**
        CONTENT_TYPE: text/x-python
        ```
        code
        ```
    and duplicated/prose-wrapped output. It does not decide validity; it only
    extracts candidate sections for strict validation later.
    """
    files: List[GeneratedFile] = []
    starts = list(SECTION_START_RE.finditer(text))
    for i, start in enumerate(starts):
        path = _clean_path(start.group("path"))
        if not path:
            continue
        body_start = start.end()
        body_end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        body = text[body_start:body_end]
        body = _drop_metadata_prefix(body)

        # Prefer explicit BEGIN/END region if present inside this section.
        begin = BEGIN_RE.search(body)
        end = END_RE.search(body, begin.end() if begin else 0) if begin else None
        if begin and end:
            body = body[begin.end():end.start()]
        else:
            # Or prefer the first fenced code block inside the section.
            fenced = re.search(r"```[A-Za-z0-9_+.-]*\s*\n(?P<content>[\s\S]*?)\n\s*```", body)
            if fenced:
                body = fenced.group("content")
            else:
                body = _trim_prose_footer(body)

        content = _clean_file_content(path, body)
        if path and content:
            files.append(GeneratedFile(path=path, content=content))
    return files


def _extract_json_files(text: str) -> List[GeneratedFile]:
    files: List[GeneratedFile] = []
    for data in _iter_json_values(text):
        possible = None
        if isinstance(data, dict):
            if isinstance(data.get("files"), list):
                possible = data["files"]
            elif data.get("path") and data.get("content"):
                possible = [data]
        elif isinstance(data, list):
            possible = data

        if not possible:
            continue

        for item in possible:
            if not isinstance(item, dict):
                continue
            path = _clean_path(str(item.get("path") or ""))
            content = _clean_file_content(path, str(item.get("content") or ""))
            if path and content:
                files.append(GeneratedFile(path=path, content=content))
    return files


def _iter_json_values(text: str) -> Iterable[Any]:
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(text) if ch in "[{" ]
    for start in starts:
        try:
            value, _ = decoder.raw_decode(text[start:])
            yield value
        except json.JSONDecodeError:
            continue


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output")
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    raise ValueError("Unclosed JSON object in model output")


def _clean_path(path: str) -> str:
    path = clean_model_text(path).strip()
    path = path.strip("`'\" *_#>")
    path = path.rstrip(":")
    match = VALID_PATH_RE.search(path)
    return match.group(0) if match else ""


def _clean_content(content: str) -> str:
    content = clean_model_text(content or "")

    # If prose wrapped a single fenced code block, prefer the fenced code over
    # trying to clean the whole prose body. This is critical for repair paths.
    fenced = _first_fenced_code_block(content)
    if fenced:
        content = fenced

    content = content.strip("\n")
    lines = content.splitlines()

    # Remove misplaced markdown fences, language labels, and content delimiters
    # repeatedly because local models often nest them incorrectly.
    changed = True
    while changed and lines:
        changed = False
        while lines and _is_discardable_prefix(lines[0]):
            lines.pop(0)
            changed = True
        while lines and _is_discardable_suffix(lines[-1]):
            lines.pop()
            changed = True

    # Remove common prose wrappers that accidentally landed inside content.
    while lines and (not lines[0].strip() or _looks_like_intro_prose(lines[0])):
        lines.pop(0)
    while lines and (not lines[-1].strip() or _looks_like_footer_prose(lines[-1])):
        lines.pop()

    # A prose-removal pass may expose markdown fences; strip again.
    changed = True
    while changed and lines:
        changed = False
        while lines and (not lines[0].strip() or _is_discardable_prefix(lines[0])):
            lines.pop(0)
            changed = True
        while lines and (not lines[-1].strip() or _is_discardable_suffix(lines[-1])):
            lines.pop()
            changed = True

    return "\n".join(lines).strip("\n")


def _clean_file_content(path: str, content: str) -> str:
    cleaned = _clean_content(content)
    if path.endswith('.py'):
        cleaned = _sanitize_python_transport_corruption(cleaned)
    return cleaned


def _first_fenced_code_block(text: str) -> str:
    """Return the largest useful fenced code block from prose output.

    Runtime repair prompts sometimes ignore FILE envelopes and return:
        Here is the repaired code:
        ```
        ...python...
        ```

    This helper extracts the code so validators never see prose/backticks.
    """
    matches = list(re.finditer(r"```[A-Za-z0-9_+.-]*\s*\n(?P<content>[\s\S]*?)\n\s*```", text or ""))
    if not matches:
        return ""
    best = max(matches, key=lambda m: _score_content(m.group('content')))
    return best.group('content')


def _sanitize_python_transport_corruption(code: str) -> str:
    """Repair transport-level corruption without deciding program logic.

    This pass only fixes obvious transport artifacts: cursor-control leftovers,
    accidental line breaks inside quoted strings, and split Python clauses like
    "as\nname:" caused by local stream corruption. The validator remains the hard
    correctness gate after this pass.
    """
    lines = code.splitlines()
    lines = _join_obvious_split_python_lines(lines)
    repaired: List[str] = []
    i = 0
    while i < len(lines):
        current = lines[i]
        guard = 0
        while _line_has_unclosed_quote(current) and i + 1 < len(lines) and guard < 8:
            i += 1
            current = current.rstrip() + " " + lines[i].lstrip()
            guard += 1
        repaired.append(current)
        i += 1
    text = "\n".join(repaired)
    text = _remove_literal_escape_fragments(text)
    return text.strip("\n")


def _join_obvious_split_python_lines(lines: List[str]) -> List[str]:
    """Join transport-split Python statements into stable single lines.

    Local streamed model output often inserts a newline in the middle of a
    Python clause or function call. This pass is deliberately mechanical: it
    only joins lines when the current statement is structurally incomplete or
    the next line is an obvious continuation. Validation still decides whether
    the resulting code is acceptable.
    """
    current_lines = list(lines)

    # Run a few passes because fixing one split may expose the next one.
    for _ in range(4):
        changed = False
        joined: List[str] = []
        i = 0
        while i < len(current_lines):
            current = current_lines[i]
            if i + 1 < len(current_lines):
                nxt = current_lines[i + 1]
                current_stripped = current.rstrip()
                nxt_stripped = nxt.lstrip()
                if _should_join_python_lines(current_stripped, nxt_stripped):
                    current = current_stripped + ' ' + nxt_stripped
                    current = _collapse_duplicate_join_fragments(current)
                    i += 1
                    changed = True
            joined.append(current)
            i += 1
        current_lines = joined
        if not changed:
            break

    return [_collapse_duplicate_join_fragments(line) for line in current_lines]


def _should_join_python_lines(current: str, nxt: str) -> bool:
    if not current or not nxt:
        return False

    current_l = current.lstrip()
    nxt_l = nxt.lstrip()

    # Do not join across normal block boundaries.
    if nxt_l.startswith(('def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except ', 'finally:', 'return ')):
        return False

    # Transport corruption often splits: open(...) as\nhandle:
    if nxt_l.startswith('as ') or nxt_l == 'as' or (nxt_l.endswith(':') and current.endswith(' as')):
        return True

    # Join if delimiters are unbalanced on the current accumulated statement.
    if _delimiter_balance(current) > 0:
        return True

    # Join if the previous line ends in a known continuation token.
    if current.endswith((',', '+', '.', '(', '[', '{', '=', '==', '!=', '<=', '>=')):
        return True

    # Join likely accidental duplicate/continuation of a function-call arg.
    if re.match(r'^[A-Za-z_][A-Za-z0-9_\.]*\)?\s*$', nxt_l) and '(' in current and ')' not in current:
        return True

    return False


def _delimiter_balance(line: str) -> int:
    """Return rough bracket/paren balance ignoring quoted strings."""
    balance = 0
    quote = None
    escaped = False
    for ch in line:
        if escaped:
            escaped = False
            continue
        if quote:
            if ch == '\\':
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in {'"', "'"}:
            quote = ch
            continue
        if ch in '([{':
            balance += 1
        elif ch in ')]}':
            balance -= 1
    return balance


def _collapse_duplicate_join_fragments(line: str) -> str:
    """Clean duplicate fragments created by cursor-left stream artifacts."""
    # foo foo) -> foo)
    line = re.sub(r'\b([A-Za-z_][A-Za-z0-9_]*)\s+\1(?=\s*[),])', r'\1', line)
    # date datetime.date -> datetime.date; common after cursor corrections.
    line = re.sub(r'\bdate\s+datetime\.', 'datetime.', line)
    line = re.sub(r'\bdateti\s+datetime\.', 'datetime.', line)
    line = re.sub(r'\batetime\s+datetime\.', 'datetime.', line)
    # expected_ou expected_output -> expected_output
    line = re.sub(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\s+\1([A-Za-z0-9_]+)', r'\1\2', line)
    return line

def _line_has_unclosed_quote(line: str) -> bool:
    state = _quote_state(line)
    return state is not None


def _quote_state(line: str) -> str | None:
    quote = None
    triple = False
    escaped = False
    i = 0
    while i < len(line):
        ch = line[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\" and quote and not triple:
            escaped = True
            i += 1
            continue
        if quote:
            if triple and line.startswith(quote * 3, i):
                quote = None
                triple = False
                i += 3
                continue
            if not triple and ch == quote:
                quote = None
                i += 1
                continue
            i += 1
            continue
        if ch in {'"', "'"}:
            # Ignore quotes in comments before a string starts.
            before = line[:i]
            if '#' in before and before.rfind('#') > before.rfind('"') and before.rfind('#') > before.rfind("'"):
                return None
            quote = ch
            triple = line.startswith(ch * 3, i)
            i += 3 if triple else 1
            continue
        i += 1
    return quote if quote else None


def _remove_literal_escape_fragments(text: str) -> str:
    # Any CSI sequences not caught earlier are removed here too.
    text = ANSI_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    return text


def _drop_metadata_prefix(body: str) -> str:
    lines = body.lstrip("\n").splitlines()
    while lines:
        stripped = lines[0].strip()
        if not stripped:
            lines.pop(0)
            continue
        if re.match(r"^[A-Z_][A-Z0-9_ -]*\s*:\s*", stripped):
            lines.pop(0)
            continue
        break
    return "\n".join(lines)


def _trim_prose_footer(body: str) -> str:
    markers = [
        "These files should",
        "Please note",
        "This provides",
        "The above",
        "You can modify",
        "RAW OUTPUT:",
    ]
    cut = len(body)
    for marker in markers:
        idx = body.find(marker)
        if idx >= 0:
            cut = min(cut, idx)
    return body[:cut]


def _is_discardable_prefix(line: str) -> bool:
    s = line.strip()
    return (
        s in {"---BEGIN CONTENT---", "<CODE>"}
        or s.startswith("```")
        or s.lower() in {"python", "py", "python3"}
    )


def _is_discardable_suffix(line: str) -> bool:
    s = line.strip()
    return s in {"---END CONTENT---", "</CODE>", "```"}


def _looks_like_intro_prose(line: str) -> bool:
    s = line.strip().lower()
    return s.startswith(("here is", "here are", "below is", "file contents", "content:", "i'll", "i will", "this code", "the repaired"))


def _looks_like_footer_prose(line: str) -> bool:
    s = line.strip().lower()
    return s.startswith(("these files", "please note", "you can", "the above", "this file"))


def _dedupe_files(candidates: List[Tuple[str, GeneratedFile]]) -> Tuple[List[GeneratedFile], List[str]]:
    by_path: Dict[str, Tuple[str, GeneratedFile]] = {}
    order: List[str] = []
    for source, item in candidates:
        if not item.path or not item.content:
            continue
        if item.path not in order:
            order.append(item.path)
        previous = by_path.get(item.path)
        if previous is None or _score_content(item.content) >= _score_content(previous[1].content):
            by_path[item.path] = (source, item)

    files = [by_path[path][1] for path in order if path in by_path]
    sources = []
    for path in order:
        if path in by_path and by_path[path][0] not in sources:
            sources.append(by_path[path][0])
    return files, sources


def _score_content(content: str) -> int:
    # Prefer content that looks like actual code and does not include prose/fences.
    score = len(content)
    if "```" in content:
        score -= 500
    if "---BEGIN CONTENT---" in content or "---END CONTENT---" in content:
        score -= 500
    if "import " in content:
        score += 100
    if "def " in content or "class " in content:
        score += 100
    return score


def _emulate_common_ansi(text: str) -> str:
    """Best-effort terminal CSI handling for local model stream artifacts.

    Supports cursor-left (D) and erase-to-end-of-line (K). Other CSI sequences
    are ignored. This is intentionally simple and deterministic.
    """
    lines: List[List[str]] = [[]]
    cursor = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\x1b" and i + 1 < len(text) and text[i + 1] == "[":
            j = i + 2
            while j < len(text) and not ("@" <= text[j] <= "~"):
                j += 1
            if j < len(text):
                params = text[i + 2:j]
                final = text[j]
                nums = [int(x) for x in re.findall(r"\d+", params)]
                n = nums[0] if nums else 1
                if final == "D":
                    cursor = max(0, cursor - n)
                elif final == "K":
                    del lines[-1][cursor:]
                i = j + 1
                continue
        if ch == "\n":
            lines.append([])
            cursor = 0
            i += 1
            continue
        line = lines[-1]
        if cursor < len(line):
            line[cursor] = ch
        else:
            line.extend(" " for _ in range(cursor - len(line)))
            line.append(ch)
        cursor += 1
        i += 1
    return "\n".join("".join(line) for line in lines)


def normalize_single_file(raw: str, expected_path: str) -> GeneratedFile:
    """Normalize one expected file from messy LLM output.

    The preferred path is to reuse the general multi-file normalizer and select
    the requested file. If the model returned only raw code or a malformed
    one-file response, fall back to extracting/cleaning the whole relevant body
    as the expected file. Validation still happens after this function.
    """
    expected_path = _clean_path(expected_path) or expected_path.strip()
    text = clean_model_text(raw or "")

    try:
        build_files = normalize_generated_files(text)
        matches = [item for item in build_files.files if item.path == expected_path]
        if matches:
            # Prefer the richest candidate for that exact path.
            matches.sort(key=lambda item: _score_content(item.content), reverse=True)
            return GeneratedFile(path=expected_path, content=matches[0].content)
    except Exception:
        pass

    # If the expected path appears in the output, take content after that header.
    header_match = None
    for match in SECTION_START_RE.finditer(text):
        if _clean_path(match.group("path")) == expected_path:
            header_match = match
            break

    if header_match:
        body = text[header_match.end():]
        next_header = SECTION_START_RE.search(body)
        if next_header:
            body = body[:next_header.start()]
        body = _drop_metadata_prefix(body)
        begin = BEGIN_RE.search(body)
        end = END_RE.search(body, begin.end() if begin else 0) if begin else None
        if begin and end:
            body = body[begin.end():end.start()]
        else:
            fenced = re.search(r"```[A-Za-z0-9_+.-]*\s*\n(?P<content>[\s\S]*?)\n\s*```", body)
            if fenced:
                body = fenced.group("content")
            else:
                body = _trim_prose_footer(body)
        content = _clean_file_content(expected_path, body)
        if content:
            return GeneratedFile(path=expected_path, content=content)

    # If the response contains a lone fenced code block, use that instead of
    # letting prose/backticks reach ast.parse. This commonly occurs in runtime
    # repair even when generation used FILE envelopes correctly.
    fenced = _first_fenced_code_block(text)
    if fenced:
        content = _clean_file_content(expected_path, fenced)
        if content:
            return GeneratedFile(path=expected_path, content=content)

    # Last resort: treat the whole response as code for this one file after
    # dropping obvious wrappers. This is useful when the prompt worked and the
    # model emitted only raw Python despite being asked for a FILE block.
    content = _clean_file_content(expected_path, text)
    return GeneratedFile(path=expected_path, content=content)
