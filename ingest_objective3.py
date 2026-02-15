"""Ingest Objective 3 corpus from sanitized files into V3 only.

This wrapper keeps Objective 3 ingestion simple and safe by:
- Reading only *.sanitized.txt files from a source directory
- Ingesting into vertical V3
- Reusing preprocess.py behavior for update/replace semantics
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import preprocess


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest sanitized Objective 3 files into V3 only"
    )
    parser.add_argument(
        "--dir",
        default="sanitized_data",
        help="Directory containing sanitized files (default: sanitized_data)",
    )
    parser.add_argument(
        "--pattern",
        default="*.sanitized.txt",
        help="Filename pattern to include (default: *.sanitized.txt)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively",
    )
    parser.add_argument(
        "--storage-path",
        default="uploaded_docs",
        help="Local document storage directory (default: uploaded_docs)",
    )
    parser.add_argument(
        "--vector-path",
        default="chroma_db",
        help="Local Chroma persist directory (default: chroma_db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print files that would be ingested",
    )
    return parser


def _collect_files(source_dir: Path, pattern: str, recursive: bool) -> List[str]:
    iterator = source_dir.rglob(pattern) if recursive else source_dir.glob(pattern)
    return [str(p.resolve()) for p in sorted(iterator) if p.is_file()]


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    source_dir = Path(args.dir).expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"{source_dir}: not found or not a directory")
        return 1

    files = _collect_files(source_dir, args.pattern, args.recursive)
    if not files:
        print(
            f"No files matched pattern '{args.pattern}' in {source_dir}. "
            "Nothing to ingest."
        )
        return 1

    if args.dry_run:
        print("Dry run. Matched files:")
        for f in files:
            print(f"- {f}")
        return 0

    print(f"Ingesting {len(files)} sanitized file(s) into V3...")
    return preprocess.main(
        [
            "--vertical",
            "V3",
            "--files",
            *files,
            "--storage-path",
            args.storage_path,
            "--vector-path",
            args.vector_path,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

