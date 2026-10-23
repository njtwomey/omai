"""Shared settings for the oracle: paths, read from flags, environment variables (WHITEBOX_*) and the root `.env`."""

from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]


class Paths(BaseModel):
    """Where case modules live and where fixtures are written."""

    cases_dir: Path = Field(default=ROOT / "oracle" / "cases", description="Directory of case modules.")
    fixtures_dir: Path = Field(
        default=ROOT / "packages" / "engine" / "fixtures", description="Fixture output directory."
    )
    only: str = Field(default="", description="Run only case modules whose relative path starts with this prefix.")
