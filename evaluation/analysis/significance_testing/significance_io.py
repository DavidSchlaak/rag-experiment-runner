"""Read local result tables for significance analyses."""

import csv
import io
from pathlib import Path
import pandas as pd

FOLDER_MIME = "application/vnd.google-apps.folder"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def find_project_root(start=None):
    current = Path(start or Path(__file__).resolve().parent).resolve()
    for candidate in (current, *current.parents):
        if all((candidate / f"{dataset}_dataset" / "results").is_dir()
               for dataset in ("tender", "ledgar", "hupd")):
            return candidate
    raise FileNotFoundError("Could not locate experiment_runner_final above the script directory.")


def _inside_project(path, project_root=PROJECT_ROOT):
    root = Path(project_root).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Refusing to access a path outside experiment_runner_final: {resolved}")
    return resolved


def list_local_children(folder_path, project_root=PROJECT_ROOT):
    """Return immediate files and folders in a Drive-compatible record shape."""
    folder = _inside_project(folder_path, project_root)
    if not folder.is_dir():
        raise FileNotFoundError(f"Expected results folder does not exist: {folder}")
    children = []
    for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
        path = _inside_project(path, project_root)
        children.append({
            "id": str(path),
            "name": path.name,
            "mimeType": FOLDER_MIME if path.is_dir() else "application/octet-stream",
            "parents": [str(path.parent)],
            "size": path.stat().st_size if path.is_file() else None,
        })
    return children


def repair_packed_csv_excel(dataframe, source_name):
    """Repair legacy XLSX files that store a CSV table in one cell."""
    df = dataframe.copy()
    meaningful_columns = [column for column in df.columns
                          if not str(column).startswith("Unnamed:")]
    if len(meaningful_columns) != 1:
        return df
    packed_column = meaningful_columns[0]
    packed_header = str(packed_column).strip()
    if "," not in packed_header:
        return df
    header = next(csv.reader([packed_header]))
    header = [str(value).strip().lstrip("\ufeff") for value in header]
    required = {"dataset", "method", "top_k", "generator_model"}
    if not required.issubset(set(header)):
        return df
    parsed_rows = []
    for row_number, value in enumerate(df[packed_column], start=2):
        if pd.isna(value):
            continue
        parsed = next(csv.reader([str(value)]))
        if len(parsed) != len(header):
            raise RuntimeError(f"Malformed packed row in {source_name}, row {row_number}.")
        parsed_rows.append(parsed)
    return pd.DataFrame(parsed_rows, columns=header)


def read_local_dataframe(file_info, project_root=PROJECT_ROOT):
    """Read a local CSV/XLSX result file without changing the source."""
    path = _inside_project(file_info["id"], project_root)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return repair_packed_csv_excel(pd.read_excel(path), path.name)
    if suffix == ".csv":
        text = path.read_bytes().decode("utf-8-sig", errors="replace")
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")
    raise ValueError(f"Unsupported file type: {path.name}")


def make_local_readers(project_root):
    """Bind the shared input readers to a selected project root."""
    root = Path(project_root).resolve()
    return (
        lambda folder_path: list_local_children(folder_path, root),
        lambda file_info: read_local_dataframe(file_info, root),
        FOLDER_MIME,
    )
