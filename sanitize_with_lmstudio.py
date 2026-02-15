"""Batch sanitize local files using a locally hosted LM Studio model."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import requests
from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".txt", ".md", ".docx", ".pdf", ".csv", ".json", ".xml", ".yaml", ".yml"}
TEXT_ENCODINGS = ["utf-8", "latin-1", "cp1252"]

ENTITY_PROMPT = """Extract identifying names from the input text.
Return JSON only with this exact schema:
{"person_names": ["..."], "org_names": ["..."]}
Rules:
- person_names: full names of individual people.
- org_names: company, organization, client, institution, fund, or business unit names.
- Use exact text spans from the input.
- Do not include generic nouns, job titles, places, months, or dates.
- If none exist, return empty arrays.
No prose. No markdown."""

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\w)"
)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
LONG_ID_RE = re.compile(r"\b(?:[A-Z]{1,6}[-#]?\d{3,}|\d{9,})\b")


@dataclass
class FileResult:
    file_id: str
    source_path: str | None
    output_path: str | None
    status: str
    chunks: int
    entities_people: int
    entities_orgs: int
    chars_in: int
    chars_out: int
    elapsed_seconds: float
    error: str | None = None


def _read_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: List[str] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"[Page {idx}]\n{text}")
    return "\n\n".join(pages)


def _read_text_bytes(raw: bytes, filename: str) -> str:
    for encoding in TEXT_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {filename} as text")


def read_file_content(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    return _read_text_bytes(path.read_bytes(), path.name)


def split_text(text: str, max_chars: int, overlap: int) -> List[str]:
    if not text.strip():
        return [""]

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            search_start = start + int(max_chars * 0.6)
            for sep in ("\n\n", "\n", ". "):
                split_idx = text.rfind(sep, search_start, end)
                if split_idx != -1:
                    end = split_idx + len(sep)
                    break

        if end <= start:
            end = min(start + max_chars, n)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks or [text]


def regex_scrub(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = URL_RE.sub("[URL]", text)
    text = SSN_RE.sub("[ID]", text)
    text = IP_RE.sub("[ID]", text)
    text = LONG_ID_RE.sub("[ID]", text)
    return text


def _clean_entity(name: str) -> str | None:
    value = name.strip().strip("\"'`[](){}<>.,;:")
    if len(value) < 2:
        return None
    lower = value.lower()
    blocked = {
        "none",
        "n/a",
        "unknown",
        "person",
        "organization",
        "org",
        "company",
    }
    if lower in blocked:
        return None
    if re.fullmatch(r"[\d\W_]+", value):
        return None
    return value


def _dedupe_entities(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for raw in values:
        cleaned = _clean_entity(raw)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return sorted(result, key=len, reverse=True)


def _extract_json_blob(text: str) -> dict:
    payload = text.strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```[a-zA-Z]*\n?", "", payload)
        payload = re.sub(r"\n?```$", "", payload)

    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        return json.loads(payload[start : end + 1])
    except json.JSONDecodeError:
        return {}


def replace_phrases(text: str, phrases: Iterable[str], replacement: str) -> str:
    output = text
    for phrase in _dedupe_entities(phrases):
        escaped = re.escape(phrase).replace(r"\ ", r"\s+")
        pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)
        output = pattern.sub(replacement, output)
    return output


class LMStudioClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int,
        retries: int,
        temperature: float,
        max_output_tokens: int,
        use_nothink: bool,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.use_nothink = use_nothink

    def list_models(self) -> List[str]:
        response = requests.get(f"{self.base_url}/v1/models", timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        return [m.get("id", "") for m in payload.get("data", []) if m.get("id")]

    def extract_entities(self, text: str) -> Tuple[List[str], List[str]]:
        user_content = f"/nothink\n{text}" if self.use_nothink else text
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role": "system", "content": ENTITY_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("Model returned non-string content")

                parsed = _extract_json_blob(content)
                people = _dedupe_entities(parsed.get("person_names", []))
                orgs = _dedupe_entities(parsed.get("org_names", []))
                return people, orgs
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2 * attempt, 10))

        raise RuntimeError(f"LM Studio request failed after {self.retries} attempts: {last_error}")


def resolve_files(input_dir: Path, explicit_files: Iterable[str] | None) -> List[Path]:
    if explicit_files:
        files = [Path(item).expanduser().resolve() for item in explicit_files]
    else:
        files = []
        for path in sorted(input_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(path.resolve())
    return files


def output_path_for(
    input_path: Path,
    input_dir: Path,
    output_dir: Path,
    index: int,
    preserve_names: bool,
) -> Path:
    if preserve_names:
        try:
            relative = input_path.resolve().relative_to(input_dir.resolve())
            filename = f"{relative.name}.sanitized.txt"
            return output_dir / relative.parent / filename
        except ValueError:
            return output_dir / f"{input_path.name}.sanitized.txt"
    return output_dir / f"file_{index:04d}.sanitized.txt"


def path_fingerprint(path: Path, input_dir: Path) -> str:
    try:
        relative = str(path.resolve().relative_to(input_dir.resolve()))
    except ValueError:
        relative = str(path.resolve())
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
    return f"sha256:{digest}"


def sanitize_text(
    source_text: str,
    client: LMStudioClient,
    max_chars: int,
    overlap: int,
    passes: int,
    chunk_workers: int,
) -> Tuple[str, int, int, int]:
    text = regex_scrub(source_text)
    total_chunks = 0
    all_people: set[str] = set()
    all_orgs: set[str] = set()

    for pass_idx in range(1, passes + 1):
        chunks = split_text(text, max_chars=max_chars, overlap=overlap)
        pass_people: set[str] = set()
        pass_orgs: set[str] = set()

        print(f"  - pass {pass_idx}/{passes}: {len(chunks)} chunk(s)")
        if chunk_workers <= 1 or len(chunks) <= 1:
            for idx, chunk in enumerate(chunks, start=1):
                print(f"    - chunk {idx}/{len(chunks)}")
                people, orgs = client.extract_entities(chunk)
                pass_people.update(people)
                pass_orgs.update(orgs)
                total_chunks += 1
        else:
            with ThreadPoolExecutor(max_workers=chunk_workers) as executor:
                futures = [executor.submit(client.extract_entities, chunk) for chunk in chunks]
                completed = 0
                for future in as_completed(futures):
                    people, orgs = future.result()
                    pass_people.update(people)
                    pass_orgs.update(orgs)
                    total_chunks += 1
                    completed += 1
                    print(f"    - chunk {completed}/{len(chunks)}")

        if not pass_people and not pass_orgs:
            break

        all_people.update(pass_people)
        all_orgs.update(pass_orgs)

        updated = replace_phrases(text, pass_orgs, "[ORG]")
        updated = replace_phrases(updated, pass_people, "[PERSON]")
        updated = regex_scrub(updated)
        if updated == text:
            break
        text = updated

    text = regex_scrub(text)
    return text, len(all_people), len(all_orgs), total_chunks


def sanitize_file(
    path: Path,
    input_dir: Path,
    output_dir: Path,
    file_id: str,
    index: int,
    preserve_names: bool,
    overwrite: bool,
    client: LMStudioClient,
    max_chars: int,
    overlap: int,
    passes: int,
    chunk_workers: int,
) -> FileResult:
    started = time.time()
    try:
        out_path = output_path_for(
            path,
            input_dir=input_dir,
            output_dir=output_dir,
            index=index,
            preserve_names=preserve_names,
        )
        if out_path.exists() and not overwrite:
            return FileResult(
                file_id=file_id,
                source_path=str(path) if preserve_names else path_fingerprint(path, input_dir=input_dir),
                output_path=str(out_path),
                status="skipped",
                chunks=0,
                entities_people=0,
                entities_orgs=0,
                chars_in=0,
                chars_out=0,
                elapsed_seconds=round(time.time() - started, 3),
            )

        source_text = read_file_content(path)
        print(f"Processing {file_id}")
        sanitized_text, people_count, org_count, chunk_count = sanitize_text(
            source_text=source_text,
            client=client,
            max_chars=max_chars,
            overlap=overlap,
            passes=passes,
            chunk_workers=chunk_workers,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(sanitized_text, encoding="utf-8")

        return FileResult(
            file_id=file_id,
            source_path=str(path) if preserve_names else path_fingerprint(path, input_dir=input_dir),
            output_path=str(out_path),
            status="ok",
            chunks=chunk_count,
            entities_people=people_count,
            entities_orgs=org_count,
            chars_in=len(source_text),
            chars_out=len(sanitized_text),
            elapsed_seconds=round(time.time() - started, 3),
        )
    except Exception as exc:  # noqa: BLE001
        return FileResult(
            file_id=file_id,
            source_path=str(path) if preserve_names else path_fingerprint(path, input_dir=input_dir),
            output_path=None,
            status="error",
            chunks=0,
            entities_people=0,
            entities_orgs=0,
            chars_in=0,
            chars_out=0,
            elapsed_seconds=round(time.time() - started, 3),
            error=str(exc),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sanitize files using a local LM Studio model")
    parser.add_argument("--input-dir", default="Data", help="Directory containing files to sanitize")
    parser.add_argument("--output-dir", default="sanitized_data", help="Directory for sanitized files")
    parser.add_argument("--files", nargs="+", help="Specific files to sanitize (overrides directory scan)")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234", help="LM Studio server base URL")
    parser.add_argument("--model", default="sanitizer", help="Model ID exposed by LM Studio")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="HTTP timeout per request")
    parser.add_argument("--retries", type=int, default=3, help="Retries per chunk")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    parser.add_argument("--max-output-tokens", type=int, default=300, help="Max completion tokens per chunk")
    parser.add_argument("--max-chars", type=int, default=4200, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=250, help="Chunk overlap in characters")
    parser.add_argument("--passes", type=int, default=2, help="Entity extraction passes per file")
    parser.add_argument(
        "--chunk-workers",
        type=int,
        default=1,
        help="Concurrent chunk requests to LM Studio per file",
    )
    parser.add_argument(
        "--use-nothink",
        action="store_true",
        help="Prefix user prompts with /nothink (for Qwen-style models)",
    )
    parser.add_argument(
        "--preserve-names",
        action="store_true",
        help="Keep original source file names/paths in outputs and report (off by default)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-sanitize files even if output already exists",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional JSON report path (defaults to <output-dir>/sanitization_report.json)",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path
        else output_dir / "sanitization_report.json"
    )

    if not input_dir.exists() or not input_dir.is_dir():
        parser.error(f"Input directory not found: {input_dir}")

    files = resolve_files(input_dir=input_dir, explicit_files=args.files)
    files = [p for p in files if p.suffix.lower() in SUPPORTED_SUFFIXES]
    if not files:
        parser.error("No supported files found to sanitize")

    client = LMStudioClient(
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        use_nothink=args.use_nothink,
    )

    try:
        available_models = client.list_models()
    except Exception as exc:  # noqa: BLE001
        parser.error(f"Could not reach LM Studio server at {args.base_url}: {exc}")

    if args.model not in available_models:
        models = ", ".join(available_models) if available_models else "<none>"
        parser.error(f"Model '{args.model}' is not available. Available models: {models}")

    print(f"Sanitizing {len(files)} file(s) with model '{args.model}' at {args.base_url}")
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Preserve names : {args.preserve_names}")
    print(f"Overwrite      : {args.overwrite}")
    print(f"Passes         : {args.passes}")
    print(f"Use /nothink   : {args.use_nothink}")
    print(f"Chunk workers  : {args.chunk_workers}")

    results: List[FileResult] = []
    for idx, path in enumerate(files, start=1):
        results.append(
            sanitize_file(
                path=path,
                input_dir=input_dir,
                output_dir=output_dir,
                file_id=f"file_{idx:04d}",
                index=idx,
                preserve_names=args.preserve_names,
                overwrite=args.overwrite,
                client=client,
                max_chars=args.max_chars,
                overlap=args.overlap,
                passes=max(1, args.passes),
                chunk_workers=max(1, args.chunk_workers),
            )
        )

    succeeded = sum(1 for item in results if item.status == "ok")
    skipped = sum(1 for item in results if item.status == "skipped")
    failed = sum(1 for item in results if item.status == "error")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "preserve_names": args.preserve_names,
        "passes": args.passes,
        "files_total": len(results),
        "files_succeeded": succeeded,
        "files_skipped": skipped,
        "files_failed": failed,
        "results": [asdict(item) for item in results],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Complete: {succeeded} succeeded, {skipped} skipped, {failed} failed")
    print(f"Report: {report_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
