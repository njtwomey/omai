"""Generate or check golden fixtures (a typed pydantic-settings CLI).

    uv run --group oracle-core python -m omai_oracle generate [--only tensor/,stats/tests/]
    uv run --group oracle-core python -m omai_oracle check [--only ...]

`--only` takes comma-separated path prefixes under oracle/cases/; a scoped check compares just those fixtures and their
manifest entries (it is what `make verify` runs for the case modules a change touches).

Each module under oracle/cases/ defines `cases() -> list[Case]` and `REFERENCE` (library name to version). Its fixture
is written to the mirrored path under packages/engine/fixtures/ with a `.json` suffix, and `manifest.json` catalogues
every fixture with its source, references and case count.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Protocol, TypedDict, cast

from pydantic_settings import BaseSettings, CliApp, CliSubCommand, SettingsConfigDict

from .fixture import Case, Fixture
from .settings import ROOT, Paths
from .vendor import rendered


class CaseModule(Protocol):
    """The contract of a module under oracle/cases/ (checked at runtime by the `Fixture` model)."""

    REFERENCE: dict[str, str]

    def cases(self) -> list[Case]: ...


class ManifestEntry(TypedDict):
    """One fixture in `manifest.json`."""

    fixture: str
    source: str
    reference: dict[str, str]
    cases: int


def load_module(path: Path) -> CaseModule:
    """Import a case module from its file path."""
    spec = importlib.util.spec_from_file_location(f"omai_cases.{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(CaseModule, module)


def prefixes(paths: Paths) -> list[str]:
    """The `--only` prefixes (empty: every case module)."""
    return [p.strip() for p in paths.only.split(",") if p.strip()]


def selected(paths: Paths, rel: Path) -> bool:
    """Whether a case module (path relative to the cases directory) is selected by `--only`."""
    chosen = prefixes(paths)
    return not chosen or any(str(rel).startswith(p) for p in chosen)


def write_fixtures(paths: Paths, out_dir: Path, existing_manifest: Path | None = None) -> list[ManifestEntry]:
    """Run every selected case module and write its fixture plus the manifest into `out_dir`.

    A partial run keeps the other modules' entries from `existing_manifest` (default: the manifest in `out_dir`).
    """
    manifest: list[ManifestEntry] = []
    for path in sorted(paths.cases_dir.rglob("*.py")):
        rel = path.relative_to(paths.cases_dir)
        if path.name.startswith("_") or not selected(paths, rel):
            continue
        module = load_module(path)
        fixture = Fixture(source=str(path.relative_to(ROOT)), reference=dict(module.REFERENCE), cases=module.cases())
        target = out_dir / rel.with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(fixture.model_dump(by_alias=True), separators=(",", ":")) + "\n")
        manifest.append(
            {
                "fixture": str(rel.with_suffix(".json")),
                "source": fixture.source,
                "reference": fixture.reference,
                "cases": len(fixture.cases),
            }
        )
    # A partial run (--only) keeps the other modules' entries of an existing manifest.
    existing = out_dir / "manifest.json"
    source = existing_manifest if existing_manifest is not None else existing
    if prefixes(paths) and source.exists():
        previous: list[ManifestEntry] = json.loads(source.read_text())["fixtures"]
        written = {m["fixture"] for m in manifest}
        manifest = sorted([m for m in previous if m["fixture"] not in written] + manifest, key=lambda m: m["fixture"])
    catalogue = {"schema": "omai.fixtures/1", "fixtures": manifest}
    existing.write_text(json.dumps(catalogue, indent=2) + "\n")
    return manifest


class Generate(Paths):
    """Regenerate fixtures and vendored datasets in place."""

    def cli_cmd(self) -> None:
        manifest = write_fixtures(self, self.fixtures_dir)
        total = sum(m["cases"] for m in manifest)
        print(f"wrote {len(manifest)} fixtures with {total} cases to {self.fixtures_dir}")
        files = {} if prefixes(self) else rendered()
        for target, text in files.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        print(f"vendored {len(files)} datasets")


class Check(Paths):
    """Regenerate fixtures into a temporary directory and fail if any committed fixture differs."""

    def cli_cmd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Scoped: the other modules' manifest entries come from the committed manifest, so it must match exactly.
            write_fixtures(self, Path(tmp), self.fixtures_dir / "manifest.json")
            stale = [
                str(p.relative_to(tmp))
                for p in sorted(Path(tmp).rglob("*.json"))
                if not (self.fixtures_dir / p.relative_to(tmp)).exists()
                or (self.fixtures_dir / p.relative_to(tmp)).read_text() != p.read_text()
            ]
        if not prefixes(self):
            stale += [
                str(t.relative_to(ROOT)) for t, text in rendered().items() if not t.exists() or t.read_text() != text
            ]
        if stale:
            print("stale fixtures or vendored datasets:", *stale, sep="\n  ")
            sys.exit(1)
        print("fixtures are up to date")


class Manifest(Paths):
    """Rebuild `manifest.json` from the fixture files already on disk, without running any case module (resolving a
    merge conflict in the manifest takes this, not a full regeneration)."""

    def cli_cmd(self) -> None:
        manifest: list[ManifestEntry] = []
        for path in sorted(self.fixtures_dir.rglob("*.json")):
            rel = path.relative_to(self.fixtures_dir)
            if rel.name == "manifest.json":
                continue
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or "cases" not in data or "source" not in data:
                continue
            fixture = cast(dict[str, object], data)
            manifest.append(
                {
                    "fixture": str(rel),
                    "source": str(fixture["source"]),
                    "reference": cast(dict[str, str], fixture.get("reference", {})),
                    "cases": len(cast(list[object], fixture["cases"])),
                }
            )
        catalogue = {"schema": "omai.fixtures/1", "fixtures": manifest}
        (self.fixtures_dir / "manifest.json").write_text(json.dumps(catalogue, indent=2) + "\n")
        print(f"manifest: {len(manifest)} fixtures")


class Oracle(BaseSettings):
    """Python ground truth for omai."""

    model_config = SettingsConfigDict(
        env_prefix="OMAI_",
        env_file=ROOT / ".env",
        extra="ignore",
        cli_prog_name="omai-oracle",
        cli_kebab_case=True,
    )

    generate: CliSubCommand[Generate]
    check: CliSubCommand[Check]
    manifest: CliSubCommand[Manifest]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    """Entry point: parse the command line and run the chosen subcommand."""
    CliApp.run(Oracle)
