"""Download model assets and unpack the externally provided project data bundle.

Run from any directory with: python setup_resources.py
The external bundle URL is configured below or through EXPERIMENT_RUNNER_ASSET_URL.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import urllib.request
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent

# set EXPERIMENT_RUNNER_ASSET_URL.
EXTERNAL_ASSET_ARCHIVE_URL = "https://unitc-my.sharepoint.com/:u:/g/personal/zxovb46_s-cloud_uni-tuebingen_de/IQDhFeHN9bFqSKVg72rETTn-AWXLUR-a-8EBcYwBZCTDKKo?e=RpzRAE"
# Optional SHA-256 for the archive. The environment variable takes precedence.
EXTERNAL_ASSET_ARCHIVE_SHA256 = ""

ALIGN_SCORE_REPO = "yzha/AlignScore"
ALIGN_SCORE_FILENAME = "AlignScore-base.ckpt"
ALIGN_SCORE_DESTINATION = PROJECT_ROOT / "evaluation" / "evaluation_runner" / "checkpoints"

# Models used by embedding preparation and metric evaluation. Hugging Face
# stores them in its standard cache, which Transformers/SentenceTransformers use.
HF_MODEL_REPOSITORIES = (
    "BAAI/bge-m3",
    "WENGSYX/Multilingual_SimCSE",
    "Siran-Li/MATCHA",
)

# External archives may restore only exact paths explicitly excluded by .gitignore.
LARGE_FILE_LIMIT_BYTES = 100 * 1024 * 1024


DIRECT_DOWNLOAD_ASSET_PATHS = {
    "evaluation/evaluation_runner/checkpoints/AlignScore-base.ckpt",
}


def listed_external_asset_paths() -> set[str]:
    """Return exact external data paths listed in .gitignore."""
    ignore_file = PROJECT_ROOT / ".gitignore"
    if not ignore_file.is_file():
        raise SystemExit(".gitignore is required to validate the external asset archive.")
    paths = {
        line.strip().lstrip("/")
        for line in ignore_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip().startswith("/") and not line.strip().startswith("//")
    }
    return paths - DIRECT_DOWNLOAD_ASSET_PATHS


def download_huggingface_assets() -> None:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing huggingface-hub. Install project dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    ALIGN_SCORE_DESTINATION.mkdir(parents=True, exist_ok=True)
    checkpoint = ALIGN_SCORE_DESTINATION / ALIGN_SCORE_FILENAME
    if checkpoint.is_file():
        print(f"AlignScore checkpoint already present: {checkpoint.relative_to(PROJECT_ROOT)}")
    else:
        print(f"Downloading {ALIGN_SCORE_REPO}/{ALIGN_SCORE_FILENAME} ...")
        hf_hub_download(
            repo_id=ALIGN_SCORE_REPO,
            filename=ALIGN_SCORE_FILENAME,
            local_dir=str(ALIGN_SCORE_DESTINATION),
        )

    for repository in HF_MODEL_REPOSITORIES:
        print(f"Checking/downloading Hugging Face model {repository} ...")
        snapshot_download(repo_id=repository)

    print("Hugging Face model assets are ready in the standard local HF cache.")


def configured_archive_url(cli_url: str | None) -> str | None:
    value = (
        cli_url
        or os.environ.get("EXPERIMENT_RUNNER_ASSET_URL")
        or EXTERNAL_ASSET_ARCHIVE_URL
    ).strip()
    if not value or value == "REPLACE_WITH_STABLE_HTTPS_DOWNLOAD_URL":
        return None
    if not value.lower().startswith("https://"):
        raise SystemExit("The external asset archive URL must use HTTPS.")
    return value


def download_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading external project data from {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "experiment-runner-setup/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def verify_archive(archive_path: Path) -> None:
    expected = (
        os.environ.get("EXPERIMENT_RUNNER_ASSET_SHA256")
        or EXTERNAL_ASSET_ARCHIVE_SHA256
    ).strip().lower()
    if not expected:
        print("No archive SHA-256 configured; skipping checksum verification.")
        return
    digest = hashlib.sha256()
    with archive_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise SystemExit(f"Asset archive SHA-256 mismatch: expected {expected}, got {actual}")
    print("External asset archive SHA-256 verified.")


def safe_extract_data_archive(archive_path: Path, overwrite: bool) -> None:
    try:
        archive = zipfile.ZipFile(archive_path)
    except zipfile.BadZipFile as exc:
        raise SystemExit("External asset URL did not return a valid ZIP archive.") from exc

    extracted = 0
    skipped = 0
    with archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise SystemExit(f"Unsafe path in data archive: {member.filename}")
            archive_name = relative.as_posix()
            if archive_name not in listed_external_asset_paths():
                raise SystemExit(
                    "The data ZIP may contain only exact external asset paths listed in .gitignore. "
                    f"Unexpected entry: {member.filename}"
                )

            # Reject Unix symlinks in ZIP files.
            unix_mode = (member.external_attr >> 16) & 0o170000
            if unix_mode == 0o120000:
                raise SystemExit(f"Symlinks are not allowed in the data archive: {member.filename}")

            destination = PROJECT_ROOT.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not overwrite:
                skipped += 1
                continue
            with archive.open(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted += 1

    print(f"External data installed: {extracted} files written, {skipped} existing files kept.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up external experiment_runner_final assets.")
    parser.add_argument(
        "--asset-url",
        help="HTTPS URL for the external data ZIP; overrides the script constant/environment variable.",
    )
    parser.add_argument(
        "--hf-only",
        action="store_true",
        help="Download Hugging Face assets only; do not require the external data ZIP.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files while extracting the external data ZIP.",
    )
    parser.add_argument("--analysis-only", action="store_true", help="Use checked-in analysis inputs; do not download any files.")
    parser.add_argument("--inventory", action="store_true", help="List local files >=100 MiB without downloading.")
    args = parser.parse_args()

    if args.inventory:
        large_files = sorted(
            (path for path in PROJECT_ROOT.rglob("*") if path.is_file() and path.stat().st_size >= 100 * 1024 * 1024),
            key=lambda path: path.stat().st_size,
            reverse=True,
        )
        for path in large_files:
            print(f"{path.stat().st_size / (1024 * 1024):8.1f} MiB  {path.relative_to(PROJECT_ROOT).as_posix()}")
        print(f"Total: {len(large_files)} files >=100 MiB")
        return 0

    if args.analysis_only:
        if args.hf_only or args.asset_url:
            parser.error("--analysis-only cannot be combined with --hf-only or --asset-url.")
        print("Analysis-only mode: no downloads performed. Use the analysis scripts with the checked-in result tables.")
        return 0

    url = configured_archive_url(args.asset_url)
    if url is None and not args.hf_only:
        print(
            "Configure EXTERNAL_ASSET_ARCHIVE_URL in setup_resources.py, set "
            "EXPERIMENT_RUNNER_ASSET_URL, or pass --asset-url.\n"
            "See EXTERNAL_ASSETS.md for the required ZIP contents and folder layout. "
            "Use --hf-only to fetch Hugging Face assets before the data URL is ready.",
            file=sys.stderr,
        )
        return 2

    download_huggingface_assets()
    if args.hf_only:
        return 0

    archive_path = PROJECT_ROOT / ".cache" / "setup_resources" / "experiment_runner_assets.zip"
    try:
        download_archive(url, archive_path)
        verify_archive(archive_path)
        safe_extract_data_archive(archive_path, overwrite=args.overwrite)
    finally:
        archive_path.unlink(missing_ok=True)
        try:
            archive_path.parent.rmdir()
            archive_path.parent.parent.rmdir()
        except OSError:
            pass
    print("Setup complete. See README.md for environment and hardware requirements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

