"""Helpers for the datasets vendored from the UCI Machine Learning Repository (CC BY 4.0): pinned downloads of the
repository's zip archives, parsing of the published files, and a reproducible stratified subsample."""

import hashlib
import io
import zipfile
from collections.abc import Callable, Sequence

from .vendor import fetch_pinned

BASE = "https://archive.ics.uci.edu/static/public"


def archive_url(uci_id: int, slug: str) -> str:
    """The URL of a dataset's zip archive."""
    return f"{BASE}/{uci_id}/{slug}.zip"


def source_note(uci_id: int, slug: str, sha256: str, members: Sequence[str]) -> str:
    """The provenance line written at the top of a generated file."""
    names = ", ".join(members)
    return f"UCI dataset {uci_id} ({archive_url(uci_id, slug)}, sha256 {sha256[:12]}; {names})"


def zip_member(uci_id: int, slug: str, sha256: str, member: str) -> bytes:
    """One file of a UCI archive, the archive pinned by its SHA-256."""
    with zipfile.ZipFile(io.BytesIO(fetch_pinned(archive_url(uci_id, slug), sha256))) as archive:
        return archive.read(member)


def nested_zip_member(uci_id: int, slug: str, sha256: str, inner: str, member: str) -> bytes:
    """A file of a zip archive inside a UCI archive."""
    with zipfile.ZipFile(io.BytesIO(zip_member(uci_id, slug, sha256, inner))) as archive:
        return archive.read(member)


def text_lines(data: bytes) -> list[str]:
    """The non-empty lines of a published file (Latin-1, any line ending), stripped."""
    return [ln.strip() for ln in data.decode("latin-1").splitlines() if ln.strip()]


def fields(line: str, sep: str | None = ",") -> list[str]:
    """The stripped fields of a line (`sep=None` splits on whitespace); a `?` is the repository's missing mark and
    becomes the empty string, which the generated files turn into `null`."""
    parts = line.split() if sep is None else line.split(sep)
    return ["" if p.strip() in ("?", "NaN") else p.strip().strip('"') for p in parts]


def table_rows(data: bytes, sep: str | None = ",", skip: int = 0) -> list[list[str]]:
    """The rows of a delimited published file as lists of fields."""
    return [fields(ln, sep) for ln in text_lines(data)[skip:]]


def columns_of(rows: Sequence[Sequence[str]], index: Sequence[int]) -> list[list[str]]:
    """The given columns of a list of rows."""
    return [[r[i] for r in rows] for i in index]


def stratified_subsample[T](labels: Sequence[T], n: int, seed: str, key: Callable[[T], str] = str) -> list[int]:
    """Row indices of a stratified subsample of `n` rows, in the original order.

    Each class keeps its share of the rows (rounded, the largest class absorbing the remainder); within a class the
    rows are ordered by the SHA-256 of `seed:index`, so the sample is the same on every machine and library version.
    """
    total = len(labels)
    classes = sorted({key(v) for v in labels})
    by_class = {c: [i for i, v in enumerate(labels) if key(v) == c] for c in classes}
    take = {c: round(n * len(rows) / total) for c, rows in by_class.items()}
    largest = max(classes, key=lambda c: len(by_class[c]))
    take[largest] += n - sum(take.values())
    chosen: list[int] = []
    for c in classes:
        ranked = sorted(by_class[c], key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
        chosen.extend(ranked[: take[c]])
    return sorted(chosen)
